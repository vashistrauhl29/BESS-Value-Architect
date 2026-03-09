import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pulp

# Set the page configuration to 'wide' layout
st.set_page_config(page_title="BESS Value Architect", layout="wide")

st.title("BESS Value Architect")
st.markdown("Evaluate the financial viability of adding battery storage to renewable energy sites based on parameters like curtailment and negative pricing.")

# --- Session State Initialization ---
if 'opt_run_complete' not in st.session_state:
    st.session_state.opt_run_complete = False
if 'financial_metrics' not in st.session_state:
    st.session_state.financial_metrics = {}
if 'optimized_data' not in st.session_state:
    st.session_state.optimized_data = None
if 'total_revenue' not in st.session_state:
    st.session_state.total_revenue = 0.0

# --- Sidebar Inputs ---
st.sidebar.header("Inputs")

st.sidebar.subheader("Technical Specifications")
power_capacity_mw = st.sidebar.number_input("Power Capacity (MW)", min_value=1.0, value=50.0, step=1.0)
duration_hours = st.sidebar.number_input("Duration (Hours)", min_value=1.0, value=4.0, step=0.5)
round_trip_efficiency = st.sidebar.number_input("Round-Trip Efficiency (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)

# Calculate energy capacity and decimal efficiency
energy_capacity_mwh = power_capacity_mw * duration_hours
efficiency_decimal = round_trip_efficiency / 100.0

st.sidebar.markdown(f"**Calculated Energy Capacity:** {energy_capacity_mwh} MWh")

st.sidebar.subheader("Financial Inputs")
capex_kwh = st.sidebar.number_input("CAPEX ($/kWh)", min_value=0.0, value=300.0, step=10.0)
opex_kw_year = st.sidebar.number_input("OPEX ($/kW-year)", min_value=0.0, value=15.0, step=1.0)
project_lifespan = st.sidebar.number_input("Project Lifespan (Years)", min_value=1, value=20, step=1)
itc_percentage = st.sidebar.number_input("ITC Percentage (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)

# --- Module A: Data Generation ---
@st.cache_data
def generate_energy_data(seed: int = 42) -> pd.DataFrame:
    """
    Simulates 8760 hours (1 year) of energy data including nodal prices,
    generation, and curtailment.
    """
    np.random.seed(seed)
    hours = np.arange(1, 8761)
    
    # Simulate Nodal Prices: Base price around $30, with some volatility
    # Introduce negative pricing events (e.g., during spring/fall nights or high wind periods)
    base_price = 30 + 20 * np.sin(2 * np.pi * hours / 24) + 15 * np.random.randn(8760)
    
    # Force some prices to be negative to simulate negative pricing events
    negative_price_mask = np.random.rand(8760) < 0.05 # 5% chance of negative prices
    nodal_price = np.where(negative_price_mask, base_price - 50 - 20 * np.random.rand(8760), base_price)
    
    # Simulate Generation: highly correlated with time of day (e.g., solar) or weather (wind)
    # We will simulate a generic renewable profile, e.g., Solar (peaks mid-day)
    daily_profile = np.maximum(0, np.sin(2 * np.pi * (hours - 6) / 24)) 
    generation = 100 * daily_profile + 10 * np.random.randn(8760)
    generation = np.maximum(0, generation) # Ensure non-negative generation
    
    # Simulate Curtailment: Correlates with high generation and negative pricing
    # If price is negative or very low, and generation is high, curtailment happens
    curtailment_factor = np.where(nodal_price < 0, 0.8, np.where(nodal_price < 10, 0.3, 0.0))
    curtailment = generation * curtailment_factor * np.random.rand(8760)
    
    df = pd.DataFrame({
        'Hour': hours,
        'Nodal_Price_$/MWh': nodal_price,
        'Generation_MWh': generation,
        'Curtailment_MWh': curtailment
    })
    return df

st.header("Module A: Energy Data Simulation")
data = generate_energy_data()
st.dataframe(data.head(24)) # Show first 24 hours as a preview

st.subheader("1-Week Energy Data Visualization (First 168 Hours)")
fig_module_a = px.line(
    data.head(168), 
    x='Hour', 
    y=['Nodal_Price_$/MWh', 'Generation_MWh', 'Curtailment_MWh'],
    title='Energy Data Overview',
    labels={'value': 'Value', 'variable': 'Metric'}
)
st.plotly_chart(fig_module_a, use_container_width=True)

# --- Module B: Optimization Engine ---
st.header("Module B: Optimization Engine")

def optimize_bess(data_subset: pd.DataFrame, power_mw: float, energy_mwh: float, efficiency: float):
    """
    Optimizes the BESS dispatch to maximize revenue for a given subset of data.
    Incorporates charging from curtailed energy at zero cost.
    """
    # Initialize the problem
    prob = pulp.LpProblem("BESS_Optimization", pulp.LpMaximize)
    
    T = len(data_subset)
    prices = data_subset['Nodal_Price_$/MWh'].values
    curtailment = data_subset['Curtailment_MWh'].values
    
    # Decision variables
    # C_grid: Charging from the grid (pays nodal price)
    C_grid = pulp.LpVariable.dicts("Charge_Grid", range(T), lowBound=0, upBound=power_mw, cat='Continuous')
    # C_curt: Charging from curtailed energy (costs $0)
    C_curt = pulp.LpVariable.dicts("Charge_Curt", range(T), lowBound=0, upBound=power_mw, cat='Continuous')
    
    D = pulp.LpVariable.dicts("Discharge", range(T), lowBound=0, upBound=power_mw, cat='Continuous')
    S = pulp.LpVariable.dicts("SoC", range(T), lowBound=0, upBound=energy_mwh, cat='Continuous')
    
    # Objective Function: Maximize (Discharge * Price - Charge_Grid * Price)
    # Note: C_curt does not incur a grid cost
    prob += pulp.lpSum([D[t] * prices[t] - C_grid[t] * prices[t] for t in range(T)]), "Total Revenue"
    
    # Constraints
    eff_sqrt = np.sqrt(efficiency)
    
    for t in range(T):
        # 1. Cannot charge more curtailed energy than is available
        prob += C_curt[t] <= curtailment[t], f"Curtailment_Limit_{t}"
        
        # 2. Total charge cannot exceed power capacity
        prob += C_grid[t] + C_curt[t] <= power_mw, f"Power_Limit_Charge_{t}"
        
        # 3. SoC tracking
        total_charge = C_grid[t] + C_curt[t]
        if t == 0:
            prob += S[t] == total_charge * eff_sqrt - D[t] / eff_sqrt, f"SoC_Balance_{t}"
        else:
            prob += S[t] == S[t-1] + total_charge * eff_sqrt - D[t] / eff_sqrt, f"SoC_Balance_{t}"
            
    # Solve the model
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    status = pulp.LpStatus[prob.status]
    
    if status == 'Optimal':
        # Extract variables
        charge_grid_vals = [C_grid[t].varValue for t in range(T)]
        charge_curt_vals = [C_curt[t].varValue for t in range(T)]
        discharge_vals = [D[t].varValue for t in range(T)]
        soc_vals = [S[t].varValue for t in range(T)]
        
        # Add columns to the subset dataframe
        data_opt = data_subset.copy()
        data_opt['Charge_Grid_MW'] = charge_grid_vals
        data_opt['Charge_Curtailment_MW'] = charge_curt_vals
        data_opt['Charge_MW'] = [g + c for g, c in zip(charge_grid_vals, charge_curt_vals)]
        data_opt['Discharge_MW'] = discharge_vals
        data_opt['SoC_MWh'] = soc_vals
        
        total_revenue = pulp.value(prob.objective)
        return data_opt, total_revenue, status
    else:
        # Return unmodified data if not optimal
        return data_subset.copy(), 0.0, status

# Streamlit UI integration for Module B
if st.button("Run Optimization"):
    with st.spinner("Running Optimization Engine..."):
        # Limit to 720 hours for Streamlit responsiveness
        opt_data = data.head(720).copy()
        
        optimized_data, total_revenue, status = optimize_bess(
            opt_data, 
            power_capacity_mw, 
            energy_capacity_mwh, 
            efficiency_decimal
        )
        
        if status == 'Optimal':
            # Store results in session state
            st.session_state.opt_run_complete = True
            st.session_state.optimized_data = optimized_data
            st.session_state.total_revenue = total_revenue
            
            # Calculate Financial Metrics
            extrapolated_annual_revenue = total_revenue * (8760 / 720)
            total_capex = energy_capacity_mwh * 1000 * capex_kwh
            itc_value = total_capex * (itc_percentage / 100.0)
            net_capex = total_capex - itc_value
            annual_opex = power_capacity_mw * 1000 * opex_kw_year
            annual_cash_flow = extrapolated_annual_revenue - annual_opex
            
            # NPV and Payback
            cash_flows = [-net_capex] + [annual_cash_flow] * int(project_lifespan)
            npv = sum([cf / ((1 + 0.08) ** i) for i, cf in enumerate(cash_flows)])
            payback_period = net_capex / annual_cash_flow if annual_cash_flow > 0 else 0
            
            st.session_state.financial_metrics = {
                "extrapolated_annual_revenue": extrapolated_annual_revenue,
                "total_capex": total_capex,
                "itc_value": itc_value,
                "net_capex": net_capex,
                "annual_opex": annual_opex,
                "annual_cash_flow": annual_cash_flow,
                "npv": npv,
                "payback_period": payback_period,
                "cash_flows": cash_flows
            }
            
            st.success(f"Solver Status: {status}")
        else:
            st.warning(f"Optimization did not find an optimal solution. Status: {status}")
            st.session_state.opt_run_complete = False

# Display Optimization Results if successful
if st.session_state.opt_run_complete:
    optimized_data = st.session_state.optimized_data
    
    st.metric("Total Optimized Revenue (First 720 Hours)", f"${st.session_state.total_revenue:,.2f}")
    
    st.subheader("BESS State of Charge (SoC) - First 720 Hours")
    fig_soc = px.line(
        optimized_data, 
        x='Hour', 
        y='SoC_MWh', 
        title='Battery State of Charge (MWh)',
        labels={'SoC_MWh': 'State of Charge (MWh)'}
    )
    st.plotly_chart(fig_soc, use_container_width=True)
    
    with st.expander("View Optimization Results Data"):
        st.dataframe(optimized_data)
else:
    st.info("Click 'Run Optimization' to determine the optimal dispatch strategy for the BESS (analyzing the first 720 hours).")

# --- Module C: Financial Outputs ---
st.header("Module C: Financial Outputs")

if st.session_state.opt_run_complete:
    st.subheader("Financial Metrics")
    metrics = st.session_state.financial_metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Net CAPEX", f"${metrics['net_capex']:,.2f}")
        st.metric("Extrapolated Annual Revenue", f"${metrics['extrapolated_annual_revenue']:,.2f}")
        
    with col2:
        st.metric("Annual Cash Flow", f"${metrics['annual_cash_flow']:,.2f}")
        st.metric("Annual OPEX", f"${metrics['annual_opex']:,.2f}")
        
    with col3:
        st.metric("Net Present Value (NPV @ 8%)", f"${metrics['npv']:,.2f}")
        st.metric("Payback Period", f"{metrics['payback_period']:.2f} Years")
        
    st.subheader("Cumulative Cash Flow Over Project Lifespan")
    cumulative_cash_flow = np.cumsum(metrics['cash_flows'])
    
    # Create DataFrame for plotting
    cf_df = pd.DataFrame({
        'Year': range(len(cumulative_cash_flow)),
        'Cumulative Cash Flow': cumulative_cash_flow
    })
    
    fig_cf = px.bar(
        cf_df, 
        x='Year', 
        y='Cumulative Cash Flow',
        title=f'Cumulative Cash Flow ({project_lifespan} Years)',
        labels={'Cumulative Cash Flow': 'Cumulative Cash Flow ($)'}
    )
    
    # Add a horizontal line at 0 for reference
    fig_cf.add_hline(y=0, line_dash="dash", line_color="red")
    
    st.plotly_chart(fig_cf, use_container_width=True)
else:
    st.info("Run the optimization in Module B to generate financial outputs.")
