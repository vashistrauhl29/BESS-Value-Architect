# BESS Value Architect: Renewable Energy Storage Optimization

## Overview
BESS Value Architect is an interactive application designed to evaluate the financial viability of co-locating Battery Energy Storage Systems (BESS) at renewable energy generation sites. The tool utilizes linear optimization to simulate charge and discharge strategies, maximizing arbitrage revenue and capturing curtailed energy.

## Business Value
Renewable energy sites often face challenges such as negative nodal pricing and grid curtailment, leading to lost revenue potential. This tool addresses these problems by intelligently dispatching a BESS to:
*   Charge during periods of negative pricing or curtailment (storing energy at zero or negative cost).
*   Discharge during periods of high nodal pricing.

By optimizing this operational profile, the application accurately calculates the total optimized revenue, Net Present Value (NPV), and Payback Period for a BESS investment.

## Technical Architecture
*   **Frontend:** Streamlit
*   **Optimization Engine:** PuLP (Linear Programming)
*   **Data Visualization:** Plotly
*   **Data Handling:** Pandas, NumPy

## Optimization Logic
The core optimization engine evaluates the optimal dispatch strategy for the battery across the operational timeline.

*   **Objective Function:** Maximize Total Revenue 
    `∑ (Discharge * Nodal Price - Grid Charge * Nodal Price)`
*   **Key Constraints:**
    *   **Power Capacity:** Total charge (Grid + Curtailment) and discharge cannot exceed the BESS MW rating.
    *   **Energy Capacity (SoC Tracking):** The State of Charge (SoC) is tracked hourly, bounded by the maximum MWh capacity, and accounts for round-trip efficiency.
    *   **Curtailment Capture:** The battery can charge from curtailed energy up to the available curtailed amount per hour at effectively $0 cost.

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install the required dependencies:**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```
