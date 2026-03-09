import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pulp
import google.generativeai as genai

st.set_page_config(page_title="BESS Value Architect", layout="wide")

st.title("BESS Value Architect")
st.markdown("Evaluate the financial viability of adding battery storage to renewable energy sites based on parameters like curtailment and negative pricing.")

if 'opt_run_complete' not in st.session_state:
    st.session_state.opt_run_complete = False
if 'financial_metrics' not in st.session_state:
    st.session_state.financial_metrics = {}
if 'optimized_data' not in st.session_state:
    st.session_state.optimized_data = None
if 'total_revenue' not in st.session_state:
    st.session_state.total_revenue = 0.0

st.sidebar.header("Inputs")

st.sidebar.subheader("Technical Specifications")
power_capacity_mw = st.sidebar.number_input("Power Capacity (MW)", min_value=1.0, value=50.0, step=1.0)
duration_hours = st.sidebar.number_input("Duration (Hours)", min_value=1.0, value=4.0, step=0.5)
round_trip_efficiency = st.sidebar.number_input("Round-Trip Efficiency (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)

energy_capacity_mwh = power_capacity_mw * duration_hours
efficiency_decimal = round_trip_efficiency / 100.0

st.sidebar.markdown(f"**Calculated Energy Capacity:** {energy_capacity_mwh} MWh")

st.sidebar.subheader("Financial Inputs")
capex_kwh = st.sidebar.number_input("CAPEX ($/kWh)", min_value=0.0, value=300.0, step=10.0)
opex_kw_year = st.sidebar.number_input("OPEX ($/kW-year)", min_value=0.0, value=15.0, step=1.0)
project_lifespan = st.sidebar.number_input("Project Lifespan (Years)", min_value=1, value=20, step=1)
itc_percentage = st.sidebar.number_input("ITC Percentage (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)

st.sidebar.subheader("AI Integrations")
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("AI Strategy Advisor: Active (Securely Connected)")
except KeyError:
    api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Required for Module D if secrets are not configured")

@st.cache_data
def generate_energy_data(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    hours = np.arange(1, 8761)
    base_price = 30 + 20 * np.sin(2 * np.pi * hours / 24) + 15 * np.random.randn(8760)
    negative_price_mask = np.random.rand(8760) < 0.05
    nodal_price = np.where(negative_price_mask, base_price - 50 - 20 * np.random.rand(8760), base_price)
    daily_profile = np.maximum(0, np.sin(2 * np.pi * (hours - 6) / 24)) 
    generation = 100 * daily_profile + 10 * np.random.randn(8760)
    generation = np.maximum(0, generation)
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
st.dataframe(data.head(24))

st.subheader("1-Week Energy Data Visualization (First 168 Hours)")
fig_module_a = px.line(
    data.head(168), 
    x='Hour', 
    y=['Nodal_Price_$/MWh', 'Generation_MWh', 'Curtailment_MWh'],
    title='Energy Data Overview',
    labels={'value': 'Value', 'variable': 'Metric'}
)
st.plotly_chart(fig_module_a, use_container_width=True)

st.header("Module B: Optimization Engine")

def optimize_bess(data_subset: pd.DataFrame, power_mw: float, energy_mwh: float, efficiency: float):
    prob = pulp.LpProblem("BESS_Optimization", pulp.LpMaximize)
    T = len(data_subset)
    prices = data_subset['Nodal_Price_$/MWh'].values
    curtailment = data_subset['Curtailment_MWh'].values
    
    C_grid = pulp.LpVariable.dicts("Charge_Grid", range(T), lowBound=0, upBound=power_mw, cat='Continuous')
    C_curt = pulp.LpVariable.dicts("Charge_Curt", range(T), lowBound=0, upBound=power_mw, cat='Continuous')
    D = pulp.LpVariable.dicts("Discharge", range(T), lowBound=0, upBound=power_mw, cat='Continuous')
    S = pulp.LpVariable.dicts("SoC", range(T), lowBound=0, upBound=energy_mwh, cat='Continuous')
    
    prob += pulp.lpSum([D[t] * prices[t] - C_grid[t] * prices[t] for t in range(T)]), "Total Revenue"
    eff_sqrt = np.sqrt(efficiency)
    
    for t in range(T):
        prob += C_curt[t] <= curtailment[t], f"Curtailment_Limit_{t}"
        prob += C_grid[t] + C_curt[t] <= power_mw, f"Power_Limit_Charge_{t}"
        total_charge = C_grid[t] + C_curt[t]
        if t == 0:
            prob += S[t] == total_charge * eff_sqrt - D[t] / eff_sqrt, f"SoC_Balance_{t}"
        else:
            prob += S[t] == S[t-1] + total_charge * eff_sqrt - D[t] / eff_sqrt, f"SoC_Balance_{t}"
            
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]
    
    if status == 'Optimal':
        charge_grid_vals = [C_grid[t].varValue for t in range(T)]
        charge_curt_vals = [C_curt[t].varValue for t in range(T)]
        discharge_vals = [D[t].varValue for t in range(T)]
        soc_vals = [S[t].varValue for t in range(T)]
        
        data_opt = data_subset.copy()
        data_opt['Charge_Grid_MW'] = charge_grid_vals
        data_opt['Charge_Curtailment_MW'] = charge_curt_vals
        data_opt['Charge_MW'] = [g + c for g, c in zip(charge_grid_vals, charge_curt_vals)]
        data_opt['Discharge_MW'] = discharge_vals
        data_opt['SoC_MWh'] = soc_vals
        
        total_revenue = pulp.value(prob.objective)
        return data_opt, total_revenue, status
    else:
        return data_subset.copy(), 0.0, status

if st.button("Run Optimization"):
    with st.spinner("Running Optimization Engine..."):
        opt_data = data.head(720).copy()
        optimized_data, total_revenue, status = optimize_bess(opt_data, power_capacity_mw, energy_capacity_mwh, efficiency_decimal)
        
        if status == 'Optimal':
            st.session_state.opt_run_complete = True
            st.session_state.optimized_data = optimized_data
            st.session_state.total_revenue = total_revenue
            
            extrapolated_annual_revenue = total_revenue * (8760 / 720)
            total_capex = energy_capacity_mwh * 1000 * capex_kwh
            itc_value = total_capex * (itc_percentage / 100.0)
            net_capex = total_capex - itc_value
            annual_opex = power_capacity_mw * 1000 * opex_kw_year
            annual_cash_flow = extrapolated_annual_revenue - annual_opex
            
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

if st.session_state.opt_run_complete:
    st.metric("Total Optimized Revenue (First 720 Hours)", f"${st.session_state.total_revenue:,.2f}")
    fig_soc = px.line(st.session_state.optimized_data, x='Hour', y='SoC_MWh', title='Battery State of Charge (MWh)')
    st.plotly_chart(fig_soc, use_container_width=True)
    with st.expander("View Optimization Results Data"):
        st.dataframe(st.session_state.optimized_data)
else:
    st.info("Click 'Run Optimization' to determine the optimal dispatch strategy.")

st.header("Module C: Financial Outputs")

if st.session_state.opt_run_complete:
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
        
    cumulative_cash_flow = np.cumsum(metrics['cash_flows'])
    cf_df = pd.DataFrame({'Year': range(len(cumulative_cash_flow)), 'Cumulative Cash Flow': cumulative_cash_flow})
    fig_cf = px.bar(cf_df, x='Year', y='Cumulative Cash Flow', title=f'Cumulative Cash Flow ({project_lifespan} Years)')
    fig_cf.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_cf, use_container_width=True)
else:
    st.info("Run the optimization in Module B to generate financial outputs.")

st.header("Module D: AI Strategy Advisor")

if st.session_state.opt_run_complete:
    if api_key:
        if st.button("Generate Executive Summary"):
            with st.spinner("Synthesizing financial data..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    metrics = st.session_state.financial_metrics
                    prompt = f"""
                    Act as an energy strategy consultant. Analyze the following BESS financial metrics and provide a 2-paragraph executive summary on the project viability.
                    Metrics: Net CAPEX: ${metrics['net_capex']:,.2f}, Annual Cash Flow: ${metrics['annual_cash_flow']:,.2f}, NPV (8% discount): ${metrics['npv']:,.2f}, Payback Period: {metrics['payback_period']:.2f} years.
                    Context: The system optimizes revenue by charging from zero-cost curtailed renewable energy and discharging during high nodal prices, avoiding negative pricing penalties. Focus strictly on financial and strategic implications. Do not use positive framing. Maintain a factual tone.
                    """
                    response = model.generate_content(prompt)
                    st.write(response.text)
                except Exception as e:
                    st.error(f"API Error: {e}")
    else:
        st.info("Provide a Gemini API Key in the sidebar to activate the AI Strategy Advisor.")
else:
    st.info("Run the optimization in Module B first.")
