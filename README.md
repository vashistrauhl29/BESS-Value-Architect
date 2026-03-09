# BESS Value Architect: Renewable Energy Storage Optimization

## Overview
This tool evaluates the financial viability of co-locating Battery Energy Storage Systems (BESS) at renewable energy sites. It utilizes linear programming to calculate optimal charge and discharge cycles, maximizing revenue through energy arbitrage and the capture of curtailed energy.

## Business Value
Avangrid and similar renewable operators face revenue loss due to energy curtailment and negative nodal pricing. This application simulates BESS integration to absorb excess generation and discharge during profitable market hours. The tool outputs a detailed financial model including Net Present Value (NPV), annual cash flows, and payback periods.

## Technical Architecture
* **Frontend:** Streamlit
* **Optimization Engine:** PuLP (Linear Programming)
* **Data Handling:** Pandas, NumPy
* **Visualization:** Plotly
* **AI Integration:** Google Gemini API (Executive Summary Generation)

## Optimization Logic
The PuLP engine maximizes a daily revenue objective function.
* **Revenue Generation:** Discharging stored energy at the current nodal price.
* **Cost Minimization:** Charging from the grid incurs the nodal price. Charging from curtailed renewable energy incurs a cost of $0/MWh.
* **Constraints:** Battery State of Charge (SoC) tracking, round-trip efficiency losses, maximum inverter power limits, and available curtailment volumes.

## Installation & Usage
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`
