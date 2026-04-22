"""BESS Value Architect — deal-level underwriting UI.

Three-file architecture:
  app.py     : Streamlit UI (this file)
  engine.py  : LP + project finance + sensitivity
  data.py    : CAISO OASIS ingestion + Parquet cache + fallback
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import google.generativeai as genai

from data import NODES, load_energy_data
from dataclasses import replace

from engine import (
    CapitalStackInputs,
    IRAInputs,
    ProjectInputs,
    run_full_analysis,
    tornado_sensitivity,
    _finance,
)

st.set_page_config(page_title="BESS Value Architect", layout="wide")
st.title("BESS Value Architect")
st.caption(
    "Deal-level underwriting for co-located battery storage at renewable sites. "
    "Real CAISO LMP data · Full-year dispatch LP · IRA §48E capital stack · "
    "DSCR-sculpted back-leverage debt · Tornado sensitivity · IC-memo drafter."
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in [
    ("results", None),
    ("tornado", None),
    ("data_bundle", None),
    ("ic_memo_text", None),
    ("tradeoff", None),         # { "payg": {"sir": ..., "min_dscr": ...}, ... }
]:
    st.session_state.setdefault(key, default)


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Market")
    node_label = st.selectbox("ISO Node", list(NODES.keys()), index=0)
    year = st.selectbox("Data Year", [2024, 2023], index=0)
    refresh = st.checkbox(
        "Refresh from CAISO OASIS",
        value=False,
        help="Live pull from CAISO OASIS (PRC_LMP, DAM). May take 30–60 s. "
             "Falls back to bundled dataset if unreachable.",
    )

    st.header("Project")
    power_mw = st.number_input("Power (MW)", 1.0, 1000.0, 50.0, 1.0)
    duration_h = st.number_input("Duration (h)", 1.0, 12.0, 4.0, 0.5)
    rte_pct = st.number_input(
        "Round-trip efficiency (%)", 70.0, 95.0, 85.0, 1.0,
        help="Tesla Megapack 2 XL 2024 spec; Fluence Gridstack — typically 85–88%.",
    )
    capex_per_kwh = st.number_input(
        "CAPEX ($/kWh)", 100.0, 800.0, 300.0, 10.0,
        help="Lazard LCOS 2024 v9 — 4-hr LFP midpoint $290–370/kWh.",
    )
    opex_per_kw_yr = st.number_input("OPEX ($/kW-yr)", 1.0, 50.0, 15.0, 1.0)
    lifespan = st.number_input("Lifespan (yr)", 10, 25, 20, 1)
    discount_rate_pct = st.number_input(
        "Sponsor discount rate (%)", 5.0, 15.0, 8.0, 0.5,
        help="Typical renewables sponsor WACC — Lazard LCOS 2024.",
    )

    st.header("Degradation & Augmentation")
    degradation_pct = st.number_input(
        "Annual fade (%)", 0.5, 5.0, 2.0, 0.5,
        help="NREL 2023 Cost & Performance Projections — LFP typical 2%/yr.",
    )
    aug_year = st.number_input("Augmentation year", 5, 20, 10, 1)
    aug_restore_pct = st.number_input(
        "Aug restores to (% nameplate)", 70.0, 100.0, 90.0, 1.0,
    )
    aug_capex_kwh = st.number_input(
        "Aug CAPEX ($/kWh of nameplate)", 0.0, 500.0, 120.0, 10.0,
    )

    aug_funding_label = st.selectbox(
        "Augmentation funding method",
        ["Reserve account", "Pay-as-you-go", "Debt refinancing at Y10"],
        index=0,
        help=(
            "Reserve account: sponsor contributes annually Y1–Y9 to a 4.5% "
            "reserve sized to cover Y10 augmentation. Lower IRR, smoother "
            "cash flows, lender-preferred for DSRA covenants.\n"
            "Pay-as-you-go: year-10 augmentation funded entirely from "
            "year-10 equity cash. Highest Sponsor IRR on a per-dollar basis; "
            "concentrated single-year risk.\n"
            "Debt refinancing at Y10: augmentation funded by an incremental "
            "amortising loan at Y10 under the original debt terms. Highest "
            "Sponsor IRR via added leverage; increases balance-sheet risk."
        ),
    )
    _aug_method_map = {
        "Reserve account": "reserve",
        "Pay-as-you-go": "payg",
        "Debt refinancing at Y10": "debt_refi",
    }
    aug_funding_method = _aug_method_map[aug_funding_label]
    reserve_rate_pct = st.number_input(
        "Reserve earning rate (%)", 0.0, 10.0, 4.5, 0.25,
        help="Typical major-maintenance reserve earnings — Callan 2024 "
             "institutional MMF benchmarks.",
    )

    st.header("IRA §48E")
    st.caption("Base: **30%** per §48E(a)(1)(A)")
    energy_community = st.checkbox(
        "Energy Community adder (+10%)",
        help="§48E(a)(3)(A). Check DOE Energy Community Tax Credit Bonus map for site.",
    )
    domestic_content = st.checkbox(
        "Domestic Content adder (+10%)",
        help="§48E(a)(3)(B). Treasury Notice 2024-41 safe-harbor cost tables.",
    )
    st.caption("Basis reduction: **50%** of ITC value applied per §48(a)(6)")

    st.header("Capital Stack")
    te_share_net_capex_pct = st.number_input(
        "TE share of net CAPEX (%)", 25.0, 55.0, 35.0, 1.0,
        help="Partnership-flip sizing: TE invests this % × (CAPEX − ITC). "
             "2026 BESS deals typically land 30–40% (Lazard LCOS 2024 v9).",
    )
    flip_year = st.number_input(
        "TE PAYGO window end (year)", 3, 15, 5, 1,
        help="End of PAYGO cash distribution to TE. Typical: year 5 after ITC monetisation.",
    )
    te_share_pre_pct = st.number_input(
        "TE cash share during PAYGO (%)", 0.0, 99.0, 5.0, 1.0,
        help="IRS-allowable small share. Classic 99/1 flip sets this to 99% (ITC-as-capital-account model).",
    )
    te_share_post_pct = st.number_input(
        "TE cash share post-PAYGO (%)", 0.0, 50.0, 0.0, 1.0,
        help="After PAYGO window TE typically drops to ~0% (transferability-style) or 5% (flip-style).",
    )
    debt_rate_pct = st.number_input(
        "Back-leverage debt rate (%)", 3.0, 12.0, 7.0, 0.25,
        help="5-yr UST + 250 bp spread, 2024–25 market.",
    )
    debt_tenor = st.number_input("Debt tenor (yr)", 5, 20, 10, 1)
    dscr_target = st.number_input(
        "Minimum DSCR", 1.10, 2.00, 1.30, 0.05,
        help="Typical back-leverage covenant — NREL ATB 2024.",
    )
    sponsor_floor_net_pct = st.number_input(
        "Sponsor equity floor (% of net CAPEX)", 15.0, 35.0, 20.0, 1.0,
        help="Minimum sponsor skin as % of (CAPEX − ITC). If binding, the "
             "DSCR target is relaxed upward to 1.40× to reduce debt sizing. "
             "Back-leverage covenant per NREL ATB 2024.",
    )
    max_dscr_cap = st.number_input(
        "Max DSCR when floor binds", 1.30, 1.60, 1.40, 0.05,
        help="Upper bound on DSCR relaxation when the sponsor floor would "
             "otherwise be violated. 1.40× is typical.",
    )

    st.header("AI (IC Memo)")
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("Gemini API: Connected")
    except (KeyError, FileNotFoundError):
        api_key = st.text_input("Gemini API Key", type="password")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
node_code = NODES[node_label]
with st.spinner(f"Loading {node_label} {year}…"):
    bundle = load_energy_data(node_code, year, refresh=refresh)
st.session_state.data_bundle = bundle

if bundle.is_live:
    st.success(f"📡 {bundle.source}")
else:
    st.warning(f"⚠️ {bundle.source}")

# Quick market characterization
_prices_preview = bundle.df["Nodal_Price_$/MWh"].values
m1, m2, m3, m4 = st.columns(4)
m1.metric("Mean DA LMP", f"${_prices_preview.mean():.2f}/MWh")
m2.metric("P95 LMP", f"${np.percentile(_prices_preview, 95):.0f}/MWh")
m3.metric("Negative-price hrs", f"{(_prices_preview < 0).sum()}")
m4.metric("Curtailment (MWh/yr)", f"{bundle.df['Curtailment_MWh'].sum():,.0f}")


# ---------------------------------------------------------------------------
# Build input objects
# ---------------------------------------------------------------------------
project = ProjectInputs(
    power_mw=power_mw,
    duration_h=duration_h,
    rte=rte_pct / 100.0,
    capex_per_kwh=capex_per_kwh,
    opex_per_kw_yr=opex_per_kw_yr,
    lifespan_years=int(lifespan),
    discount_rate=discount_rate_pct / 100.0,
    degradation_rate=degradation_pct / 100.0,
    aug_year=int(aug_year),
    aug_restore_frac=aug_restore_pct / 100.0,
    aug_capex_per_kwh=aug_capex_kwh,
    aug_funding_method=aug_funding_method,
    reserve_rate=reserve_rate_pct / 100.0,
)
ira = IRAInputs(
    base_itc=0.30,
    energy_community=energy_community,
    domestic_content=domestic_content,
    basis_reduction_frac=0.50,
)
stack = CapitalStackInputs(
    te_share_of_net_capex=te_share_net_capex_pct / 100.0,
    te_cash_share_pre_flip=te_share_pre_pct / 100.0,
    te_cash_share_post_flip=te_share_post_pct / 100.0,
    flip_year=int(flip_year),
    debt_rate=debt_rate_pct / 100.0,
    debt_tenor=int(debt_tenor),
    dscr_target=dscr_target,
    sponsor_floor_frac_of_net_capex=sponsor_floor_net_pct / 100.0,
    max_dscr_for_sponsor_floor=max_dscr_cap,
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if st.button("Run Analysis", type="primary", use_container_width=True):
    prices = bundle.df["Nodal_Price_$/MWh"].values
    curtailment = bundle.df["Curtailment_MWh"].values
    with st.spinner("Solving 8,760-hour LP (CBC)…"):
        results = run_full_analysis(prices, curtailment, project, ira, stack)
    if results["status"] != "Optimal":
        st.error(f"LP did not reach optimality. Status: {results['status']}")
        st.session_state.results = None
    else:
        st.session_state.results = results
        # Capital Structure Tradeoff — re-run the finance layer under all
        # three funding methods from the same LP output (cheap; the LP is
        # invariant to method).
        tradeoff = {}
        for m in ("payg", "reserve", "debt_refi"):
            p_m = replace(project, aug_funding_method=m)
            r_m = _finance(results["optimization"].annual_revenue_year1, p_m, ira, stack)
            tradeoff[m] = {
                "sponsor_irr": r_m["returns"]["sponsor_irr"],
                "project_irr": r_m["returns"]["project_irr"],
                "min_dscr": r_m["returns"]["min_dscr"],
                "min_dscr_debt_only": r_m["returns"]["min_dscr_debt_only"],
            }
        st.session_state.tradeoff = tradeoff
        with st.spinner("Running tornado sensitivity…"):
            st.session_state.tornado = tornado_sensitivity(
                prices, curtailment, project, ira, stack, results["optimization"],
            )
        st.session_state.ic_memo_text = None  # invalidate old memo


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
results = st.session_state.results
if not results:
    st.info("Configure the sidebar and click **Run Analysis**.")
    st.stop()

opt = results["optimization"]
ret = results["returns"]
src = results["sources"]


# --- Summary metrics -------------------------------------------------------
st.header("Summary")
row1 = st.columns(4)
row1[0].metric("Project IRR", f"{ret['project_irr']*100:.1f}%" if np.isfinite(ret['project_irr']) else "—")
row1[1].metric("Sponsor IRR", f"{ret['sponsor_irr']*100:.1f}%" if np.isfinite(ret['sponsor_irr']) else "—")
row1[2].metric("Project NPV", f"${ret['project_npv']/1e6:,.1f} M")
row1[3].metric("Payback", f"{ret['payback_year']} yr" if ret['payback_year'] else "> life")

row2 = st.columns(4)
row2[0].metric(
    "Min DSCR (senior obligations)",
    f"{ret['min_dscr']:.2f}×" if np.isfinite(ret['min_dscr']) else "—",
    help="CFADS / (debt service + reserve contribution). The binding covenant.",
)
row2[1].metric(
    "Min DSCR (debt only)",
    f"{ret['min_dscr_debt_only']:.2f}×" if np.isfinite(ret['min_dscr_debt_only']) else "—",
    help="Conventional lender metric: CFADS / debt service. Floats above 1.30× under the Reserve method.",
)
row2[2].metric("Year-1 Revenue", f"${opt.annual_revenue_year1/1e6:.2f} M")
row2[3].metric("Year-10 Aug CAPEX", f"${results['aug_capex']/1e6:.2f} M")

# --- LP sanity metrics -----------------------------------------------------
disp = opt.dispatch
total_discharge_mwh = float(disp["Discharge_MW"].sum())
total_charge_mwh = float((disp["Charge_Grid_MW"] + disp["Charge_Curt_MW"]).sum())
neg_price_hours_absorbed = int(((disp["Nodal_Price_$/MWh"] < 0) &
                                 ((disp["Charge_Grid_MW"] + disp["Charge_Curt_MW"]) > 0.01)).sum())
row3 = st.columns(4)
row3[0].metric("Annual Cycles", f"{opt.cycles_per_year:.0f}")
row3[1].metric("Total Discharge (MWh/yr)", f"{total_discharge_mwh:,.0f}")
row3[2].metric("Total Charge (MWh/yr)", f"{total_charge_mwh:,.0f}")
row3[3].metric("Neg-price hrs absorbed", f"{neg_price_hours_absorbed}")


# --- Capital Structure Tradeoff -------------------------------------------
tradeoff = st.session_state.tradeoff
if tradeoff is not None:
    st.subheader("Capital Structure Tradeoff — Sponsor IRR by funding method")
    current_method = project.aug_funding_method
    t_cols = st.columns(3)
    method_order = [
        ("payg", "Pay-as-you-go", t_cols[0]),
        ("reserve", "Reserve account", t_cols[1]),
        ("debt_refi", "Debt refi at Y10", t_cols[2]),
    ]
    for m_key, m_label, col in method_order:
        d = tradeoff[m_key]
        sir = d["sponsor_irr"]
        sir_s = f"{sir*100:.1f}%" if np.isfinite(sir) else "—"
        dscr_s = f"Min DSCR {d['min_dscr']:.2f}× / {d['min_dscr_debt_only']:.2f}×"
        if m_key == current_method:
            col.metric(f"▶ {m_label}  (selected)", sir_s, dscr_s)
        else:
            col.caption(f"{m_label}")
            col.markdown(f"<span style='color:#888888'>{sir_s}<br><small>{dscr_s}</small></span>",
                         unsafe_allow_html=True)
    st.caption(
        "Tradeoff: leverage drives IRR. Debt-refi adds leverage at Y10 (highest IRR, "
        "highest balance-sheet risk). PAYG keeps Y10 risk concentrated on sponsor. "
        "Reserve is a risk-reduction / covenant-compliance mechanism, not an IRR "
        "enhancer — it de-levers the project in exchange for smoother cash flows."
    )


# --- Capital stack ---------------------------------------------------------
st.subheader("Capital Stack (Sources & Uses)")
stack_df = pd.DataFrame(
    {
        "Component": [
            "Total CAPEX (use)",
            "— Tax Equity Investment",
            "— Back-Leverage Debt",
            "— Sponsor Equity",
            "ITC Value (tax benefit)",
            f"ITC Rate Applied",
            "Depreciable Basis (post §48(a)(6))",
        ],
        "$M / %": [
            f"${results['capex']/1e6:,.1f} M",
            f"${src['te_investment']/1e6:,.1f} M",
            f"${src['debt_principal']/1e6:,.1f} M",
            f"${src['sponsor_equity']/1e6:,.1f} M",
            f"${results['itc_value']/1e6:,.1f} M",
            f"{results['itc_rate']*100:.0f}%",
            f"${results['depreciable_basis']/1e6:,.1f} M",
        ],
    }
)
st.dataframe(stack_df, hide_index=True, use_container_width=True)


# --- Dispatch — representative week ---------------------------------------
st.header("Dispatch — Representative Summer Week (Jul 1–7)")
wk_start = 24 * 181  # ~Jul 1
wk = opt.dispatch.iloc[wk_start:wk_start + 168].copy()

fig_disp = go.Figure()
fig_disp.add_trace(go.Bar(
    x=wk["Hour"], y=wk["Discharge_MW"], name="Discharge (MW)",
    marker_color="#2E7D32",
))
fig_disp.add_trace(go.Bar(
    x=wk["Hour"], y=-(wk["Charge_Grid_MW"] + wk["Charge_Curt_MW"]),
    name="Charge (MW)", marker_color="#EF6C00",
))
fig_disp.add_trace(go.Scatter(
    x=wk["Hour"], y=wk["SoC_MWh"], name="SoC (MWh)",
    line=dict(color="#1565C0", dash="dot"),
))
fig_disp.add_trace(go.Scatter(
    x=wk["Hour"], y=wk["Nodal_Price_$/MWh"], name="LMP ($/MWh)",
    line=dict(color="#C62828"), yaxis="y2",
))
fig_disp.update_layout(
    barmode="relative",
    yaxis=dict(title="MW / MWh"),
    yaxis2=dict(title="LMP ($/MWh)", overlaying="y", side="right"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=400,
)
st.plotly_chart(fig_disp, use_container_width=True)


# --- Annual cash flows -----------------------------------------------------
st.header("Annual Cash Flows")
years_idx = np.arange(len(results["project_cf"]))
cf_df = pd.DataFrame({
    "Year": years_idx,
    "Project CFADS (pre-finance)": results["project_cf"],
    "Tax Equity CF": results["te_cf"],
    "Sponsor CF": results["sponsor_cf"],
    "Debt Service": np.concatenate([[0.0], -results["debt_service_full"]]),
})
fig_cf = px.bar(
    cf_df.melt(id_vars="Year", var_name="Party", value_name="Cash Flow ($)"),
    x="Year", y="Cash Flow ($)", color="Party", barmode="group", height=400,
)
fig_cf.update_yaxes(tickformat="$,.0f")
st.plotly_chart(fig_cf, use_container_width=True)

with st.expander("Cash flow table"):
    st.dataframe(cf_df.style.format({c: "${:,.0f}" for c in cf_df.columns if c != "Year"}),
                 use_container_width=True)


# --- DSCR by year ----------------------------------------------------------
st.subheader("DSCR by Year (Debt Tenor)")
dscr_s = results["dscr_senior_by_year"]
dscr_d = results["dscr_debt_only_by_year"]
dscr_df = pd.DataFrame({
    "Year": np.arange(1, len(dscr_s) + 1),
    "Senior (debt + reserve)": np.where(np.isfinite(dscr_s), dscr_s, 0.0),
    "Debt only": np.where(np.isfinite(dscr_d), dscr_d, 0.0),
})
fig_dscr = px.bar(
    dscr_df.melt(id_vars="Year", var_name="DSCR type", value_name="DSCR"),
    x="Year", y="DSCR", color="DSCR type", barmode="group", height=300,
)
fig_dscr.add_hline(
    y=stack.dscr_target, line_dash="dash", line_color="red",
    annotation_text=f"Target {stack.dscr_target:.2f}×",
)
st.plotly_chart(fig_dscr, use_container_width=True)


# --- Augmentation Reserve --------------------------------------------------
st.subheader("Augmentation Reserve")
if project.aug_funding_method == "reserve":
    rb = results["reserve_balance"]
    reserve_df = pd.DataFrame({
        "Year": np.arange(len(rb)),
        "Reserve Balance ($)": rb,
    })
    fig_res = px.area(
        reserve_df, x="Year", y="Reserve Balance ($)", height=280,
        color_discrete_sequence=["#1565C0"],
    )
    fig_res.add_hline(
        y=results["aug_capex"], line_dash="dash", line_color="red",
        annotation_text=f"Aug CAPEX ${results['aug_capex']/1e6:.1f}M",
    )
    fig_res.update_yaxes(tickformat="$,.0f")
    st.plotly_chart(fig_res, use_container_width=True)
    st.info(
        f"Annual contribution: \\${results['reserve_contribution']/1e6:.2f}M "
        f"(years 1–{project.aug_year-1}). Balance reaches "
        f"\\${rb[project.aug_year-1]/1e6:.2f}M at end of year {project.aug_year-1}, "
        f"fully drawn in year {project.aug_year} to pay augmentation CAPEX."
    )
    st.caption(
        "The Augmentation Reserve is a risk-reduction mechanism, not an "
        "IRR-enhancement mechanism. It smooths sponsor cash flows, satisfies "
        "typical lender DSRA covenants, and eliminates the year-10 equity "
        "shock — at the cost of lower Sponsor IRR versus pay-as-you-go at "
        "these inputs."
    )
elif project.aug_funding_method == "debt_refi":
    st.info(
        f"Augmentation funded via a \\${results['aug_capex']/1e6:.1f}M amortising loan "
        f"at year {project.aug_year}, same rate as primary debt "
        f"({stack.debt_rate*100:.2f}%), 10-year tenor. Annual incremental debt "
        f"service: \\${results['refi_debt_service']/1e6:.2f}M."
    )
else:
    st.info(
        f"Pay-as-you-go: year {project.aug_year} sponsor equity cash absorbs "
        f"the full \\${results['aug_capex']/1e6:.1f}M augmentation outflow."
    )


# --- Tornado ---------------------------------------------------------------
tornado = st.session_state.tornado
if tornado is not None and len(tornado):
    st.header("Sensitivity — Tornado on Project NPV")
    base_npv = float(tornado["base_npv"].iloc[0])

    # Data values in $M (not raw dollars) so auto-ticks land on clean integer
    # million steps ($5M, $10M, ...) rather than colliding via SI rounding.
    base_npv_m = base_npv / 1e6
    fig_t = go.Figure()
    for _, row in tornado.iterrows():
        hi_m = (row["high_npv"] - base_npv) / 1e6
        lo_m = (row["low_npv"] - base_npv) / 1e6
        fig_t.add_trace(go.Bar(
            y=[row["variable"]], x=[hi_m], orientation="h",
            marker_color="#2E7D32", base=base_npv_m, showlegend=False,
            hovertemplate="High: $%{x:,.1f}M<extra></extra>",
        ))
        fig_t.add_trace(go.Bar(
            y=[row["variable"]], x=[lo_m], orientation="h",
            marker_color="#C62828", base=base_npv_m, showlegend=False,
            hovertemplate="Low: $%{x:,.1f}M<extra></extra>",
        ))
    fig_t.add_vline(
        x=base_npv_m, line_dash="dash", line_color="black",
        annotation_text=f"Base NPV: ${base_npv_m:,.1f}M",
    )
    fig_t.update_layout(
        xaxis_title="Project NPV ($M)",
        yaxis=dict(autorange="reversed"),
        height=420,
    )
    # Explicit integer tick format; dtick=5 enforces clean $5M spacing so
    # the axis reads $0, $5M, $10M, $15M, $20M, $25M, $30M, $35M — no
    # SI-rounding collisions.
    fig_t.update_xaxes(tickprefix="$", ticksuffix="M", tickformat=",.0f", dtick=5)
    st.plotly_chart(fig_t, use_container_width=True)

    with st.expander("Tornado table"):
        disp = tornado[["variable", "low_npv", "high_npv", "range"]].copy()
        st.dataframe(disp.style.format({
            "low_npv": "${:,.0f}", "high_npv": "${:,.0f}", "range": "${:,.0f}",
        }), use_container_width=True)
    st.caption("RTE sensitivity uses a linear revenue-scaling approximation "
               "around the base dispatch; a full re-optimisation at perturbed "
               "RTE would diverge ≤10% for |ΔRTE| ≤ 7 pp.")


# ---------------------------------------------------------------------------
# IC Memo Drafter
# ---------------------------------------------------------------------------
st.header("Investment Committee Memo")
col_a, col_b = st.columns([1, 3])
if api_key:
    draft_clicked = col_a.button("Draft IC Memo", use_container_width=True)
    col_b.caption(
        "Drafts a structured IC memo from the full analysis using Gemini 2.5 Flash: "
        "Thesis · Sources of Value · Capital Structure Summary · Key Risks · "
        "Sensitivities · Recommendation."
    )
else:
    draft_clicked = False
    col_a.button("Draft IC Memo", use_container_width=True, disabled=True)
    col_b.caption("Set `GEMINI_API_KEY` (sidebar or secrets) to enable IC memo drafting.")

if draft_clicked:
    if True:
        top3 = tornado.head(3)[["variable", "low_npv", "high_npv", "range"]].to_dict("records")
        adders = []
        if ira.energy_community:
            adders.append("Energy Community +10%")
        if ira.domestic_content:
            adders.append("Domestic Content +10%")
        adder_str = ", ".join(adders) if adders else "none"

        _method_display = {
            "reserve": "Augmentation Reserve Account (sinking-fund sweep years 1-9 at 4.5%, drawn Y10)",
            "payg": "Pay-as-you-go (sponsor absorbs full Y10 augmentation from operating cash)",
            "debt_refi": "Debt Refinancing at Y10 (new 10-year amortising loan at the primary debt rate)",
        }
        method_display = _method_display.get(project.aug_funding_method, project.aug_funding_method)

        tr = st.session_state.tradeoff or {}
        tr_line = ""
        if tr:
            tr_line = (
                f"  Cross-method Sponsor IRR: "
                f"PAYG {tr['payg']['sponsor_irr']*100:.1f}% / "
                f"Reserve {tr['reserve']['sponsor_irr']*100:.1f}% / "
                f"Debt-refi {tr['debt_refi']['sponsor_irr']*100:.1f}%"
            )

        prompt = f"""You are an energy infrastructure investment analyst drafting an Investment Committee (IC) memo. Use the exact numbers below — do not invent figures. Do not use positive framing. ~500 words total. Output in the exact Markdown structure specified.

PROJECT CONTEXT
  Node: {node_label}  ({node_code})
  Data source: {bundle.source}
  Market stats (year): mean DA LMP ${_prices_preview.mean():.2f}/MWh, P95 ${np.percentile(_prices_preview,95):.0f}/MWh, {(_prices_preview<0).sum()} negative-price hours
  Power / Duration / Energy: {project.power_mw} MW / {project.duration_h} h / {project.energy_mwh} MWh
  RTE: {project.rte*100:.0f}%   Lifespan: {project.lifespan_years} yr   Degradation: {project.degradation_rate*100:.1f}%/yr
  Year-{project.aug_year} augmentation: restore to {project.aug_restore_frac*100:.0f}% nameplate at ${results['aug_capex']/1e6:.1f}M

AUGMENTATION FUNDING — SELECTED
  Method: {method_display}
{tr_line}

CAPITAL STACK (at the selected funding method)
  Total CAPEX: ${results['capex']/1e6:.1f}M
  ITC rate: {results['itc_rate']*100:.0f}% (base 30% + adders: {adder_str})
  ITC value: ${results['itc_value']/1e6:.1f}M      Depreciable basis (§48(a)(6)): ${results['depreciable_basis']/1e6:.1f}M
  Tax Equity: ${src['te_investment']/1e6:.1f}M   (sized at {stack.te_share_of_net_capex*100:.0f}% of net CAPEX)
  Back-Leverage Debt: ${src['debt_principal']/1e6:.1f}M @ {stack.debt_rate*100:.2f}%, {stack.debt_tenor}-yr tenor, base DSCR target {stack.dscr_target:.2f}×
  Sponsor Equity: ${src['sponsor_equity']/1e6:.1f}M  (floor {stack.sponsor_floor_frac_of_net_capex*100:.0f}% of net CAPEX)

OPERATIONS (Year 1)
  Annual arbitrage + curtailment-capture revenue: ${opt.annual_revenue_year1/1e6:.2f}M
  Full-equivalent cycles: {opt.cycles_per_year:.0f}/yr
  Annual OPEX: ${results['annual_opex']/1e6:.2f}M

RETURNS
  Project IRR: {ret['project_irr']*100:.1f}%
  Sponsor IRR: {ret['sponsor_irr']*100:.1f}%
  Project NPV @ {project.discount_rate*100:.1f}%: ${ret['project_npv']/1e6:.1f}M
  Min DSCR (senior obligations): {ret['min_dscr']:.2f}×
  Min DSCR (debt only): {ret['min_dscr_debt_only']:.2f}×
  Payback: {ret['payback_year'] if ret['payback_year'] else 'beyond project life'}

TOP NPV SENSITIVITIES (sorted by range)
""" + "\n".join([
    f"  - {r['variable']}: low ${r['low_npv']/1e6:.1f}M → high ${r['high_npv']/1e6:.1f}M (range ${r['range']/1e6:.1f}M)"
    for r in top3
]) + """

OUTPUT FORMAT (use these exact Markdown headers in this order):

## Thesis
Two sentences. State the deal (size, node, revenue drivers) and whether it pencils given the returns above.

## Sources of Value
Bulleted list. For each source, give a specific $ figure. Name: (a) energy arbitrage revenue, (b) curtailment-capture (zero-fuel charging), (c) IRA §48E ITC value including any adders, (d) tax equity monetisation lift if material.

## Capital Structure Summary
Three sentences. Name the sources & uses by $ amount. Identify the selected augmentation funding method and give one sentence on why the sponsor chose it over the alternatives — compare against the cross-method Sponsor IRR line above and note the risk/IRR tradeoff.

## Key Risks
Bulleted list of the top three risks, each quantified using the sensitivity figures above. Name each risk in one phrase, then the $ downside.

## Sensitivities
Two sentences. State which two variables move NPV the most and by how much.

## Recommendation
One paragraph ending with: "Approve", "Approve with conditions:" + specific conditions, or "Decline: " + specific reason. Reference the Min DSCR and Sponsor IRR by value."""

        with st.spinner("Drafting IC memo…"):
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.5-flash")
                resp = model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.2},
                )
                st.session_state.ic_memo_text = resp.text
            except Exception as e:
                st.error(f"Gemini error: {e}")

if st.session_state.ic_memo_text:
    st.markdown("---")
    st.markdown(st.session_state.ic_memo_text.replace("$", r"\$"))
