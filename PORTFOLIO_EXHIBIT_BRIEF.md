# BESS Value Architect — Deloitte Portfolio Exhibit Brief

> Consulting-grade audit for a senior energy-sector executive reader.
> Scope: everything in the repo as of the audit date. No speculation beyond code.

---

## 1. Repository Structure

```
BESS-Value-Architect-main/
├── app.py                          # Entire application (UI + optimizer + finance + AI)
├── ping.py                         # Playwright script to keep Streamlit app warm
├── requirements.txt                # 6 packages: streamlit, pandas, numpy, plotly, pulp, google-generativeai
├── README.md                       # 26 lines, project overview
├── .devcontainer/devcontainer.json # Codespaces config (Python 3.11)
└── .github/workflows/keep_alive.yml # Cron every 6h — pings the Streamlit URL
```

**Honest framing**: there is no `docs/`, no PRD, no separate `optimizer.py` or `financial_model.py`. The entire system lives in a single **220-line Streamlit script** with four numbered modules (A → D).

### Module-to-file mapping (all in `app.py`)

| Module | Lines | Purpose |
|---|---|---|
| A. Energy Data Simulation | 47–80 | Synthetic 8,760-hour nodal price + generation + curtailment series |
| B. Optimization Engine (PuLP) | 82–172 | LP model for BESS dispatch |
| C. Financial Outputs | 174–195 | NPV, payback, cumulative cash-flow chart |
| D. AI Strategy Advisor (Gemini) | 197–219 | Executive summary generation via `gemini-2.5-flash` |

### Tech stack layers & why each exists

- **Streamlit** — zero-backend analyst tool; fits a Deloitte team prototyping with no infra.
- **PuLP + CBC** (default solver) — open-source LP; sufficient for <10k continuous variables like a 720-hour dispatch.
- **Pandas / NumPy** — vectorized data generation and results joining.
- **Plotly** — interactive SoC and cash-flow charts.
- **Google Gemini (`gemini-2.5-flash`)** — narrative layer only; it does **not** drive optimization or valuation.

---

## 2. PRD / Design Doc Search

**There is no PRD, architecture doc, design note, or `docs/` directory.** The README is the only narrative artifact.

### Quoted verbatim from README.md

> "This tool evaluates the financial viability of co-locating Battery Energy Storage Systems (BESS) at renewable energy sites."

> "Avangrid and similar renewable operators face revenue loss due to energy curtailment and negative nodal pricing. This application simulates BESS integration to absorb excess generation and discharge during profitable market hours."

> "**Revenue Generation:** Discharging stored energy at the current nodal price."

> "**Cost Minimization:** Charging from the grid incurs the nodal price. Charging from curtailed renewable energy incurs a cost of $0/MWh."

> "**Constraints:** Battery State of Charge (SoC) tracking, round-trip efficiency losses, maximum inverter power limits, and available curtailment volumes."

The "Business Value" section names **Avangrid** explicitly — framing the tool as a pitch artifact for utility-scale renewable operators, not a generic toolkit.

### Implicit BESS economic assumptions (not documented anywhere)

- Round-trip efficiency: default **85%** (user-editable, `app.py:27`).
- Project lifespan: default **20 years** (user-editable, `app.py:37`).
- Discount rate: **hardcoded 8%** in NPV calc (`app.py:146`) — not exposed as input.
- ITC percentage: default 30%, user-editable (`app.py:38`).
- **No DSCR target, no leverage ratio, no debt sizing.** 100% equity-financed.
- **No degradation, no capacity fade, no augmentation schedule.** Annual revenue is held flat across 20 years.
- **No cycle-life constraint.** The optimizer may charge/discharge at full power every hour with zero throughput penalty.

---

## 3. The Untapped Insight — Energy-Sector Specific

### What the tool solves that Aurora / PLEXOS / HOMER don't

Honestly: it does not compete with these platforms — it is a **pedagogical prototype** that reframes a narrow slice of the problem. Aurora and PLEXOS run nodal production-cost simulations over fundamental fuel and build-out assumptions; HOMER does micro-grid optimization; Excel pro formas handle the capital stack. What this tool uniquely foregrounds is:

- **Curtailed-energy-as-fuel accounting** — charging from curtailed renewables is explicitly a **$0/MWh** opportunity cost, distinct from grid charging. Enterprise tools model curtailment as an output, not as a free storage fuel-source line item.
- **Negative-LMP absorption economics** — the synthetic price series is deliberately engineered to create ~5% negative-price hours so the optimizer can demonstrate avoided-penalty value.

Everything else — the LP, NPV, ITC haircut — is standard undergraduate-level asset-finance modeling.

### The PuLP formulation, in the code's own words

**Decision variables** (`app.py:90–93`):
```python
C_grid = pulp.LpVariable.dicts("Charge_Grid", range(T), lowBound=0, upBound=power_mw, cat='Continuous')
C_curt = pulp.LpVariable.dicts("Charge_Curt", range(T), lowBound=0, upBound=power_mw, cat='Continuous')
D      = pulp.LpVariable.dicts("Discharge",    range(T), lowBound=0, upBound=power_mw, cat='Continuous')
S      = pulp.LpVariable.dicts("SoC",          range(T), lowBound=0, upBound=energy_mwh, cat='Continuous')
```

**Objective** (`app.py:95`):
```python
prob += pulp.lpSum([D[t] * prices[t] - C_grid[t] * prices[t] for t in range(T)]), "Total Revenue"
```

In plain form: **maximize Σₜ (Dₜ · pₜ − C_gridₜ · pₜ)**. Charging from curtailment has zero cost in the objective — this is how the README's $0/MWh claim is realized.

**Constraints** (`app.py:98–105`):
```python
for t in range(T):
    prob += C_curt[t] <= curtailment[t]                              # curtailment availability
    prob += C_grid[t] + C_curt[t] <= power_mw                        # inverter charge limit
    # SoC balance with split efficiency:
    prob += S[t] == S[t-1] + total_charge * eff_sqrt - D[t] / eff_sqrt
```

The **η̂ = √η** trick (`eff_sqrt = np.sqrt(efficiency)`) splits round-trip efficiency symmetrically across charge and discharge — a common convention.

### Missing constraints a senior energy reader will catch

- **No discharge-power constraint** beyond the variable upper bound (acceptable but absent as an explicit constraint).
- **No mutual exclusion** between charge and discharge — the LP could theoretically do both at once. Economically suppressed by the objective, but not structurally blocked.
- **No SoC terminal condition** (S[T-1] free → optimizer empties the battery at horizon, biasing revenue upward).
- **No daily/monthly cycle-count constraint** — cycle life is not modeled.
- **No degradation** reducing `energy_mwh` over time.

### Revenue streams actually modeled

**Only energy arbitrage + curtailment absorption.** Not modeled anywhere in code:

- Ancillary services (FRAS, spinning reserve, regulation up/down)
- Capacity payments (ISO capacity auctions, RA in CAISO, ICAP in NYISO / PJM)
- Demand-charge reduction (behind-the-meter)
- Resource Adequacy / Clean Peak Standard credits
- Black-start, reactive power, voltage support

### IRA tax credits — the real state

The README and UI say "ITC Percentage". The code implementation (`app.py:140–141`):
```python
itc_value = total_capex * (itc_percentage / 100.0)
net_capex = total_capex - itc_value
```

This is a **single flat haircut on CAPEX**. There is:

- **No reference to §48E (Clean Electricity ITC), §45Y (Clean Electricity PTC), §45X, §30C, §45V, or §6418 transferability** anywhere in the code.
- **No bonus-adder logic** for energy communities (+10%), domestic content (+10%), or low-income (+10/+20%).
- **No normalization / basis reduction** (ITC basis is normally reduced by 50% of the credit — not handled).
- **No PTC alternative pathway** — storage is ITC-only in the real IRA, which is correct, but the model doesn't distinguish storage vs. solar+storage stacking.

### Dispatch output structure

Returned as a dataframe (`app.py:116–121`):
```python
data_opt['Charge_Grid_MW'], data_opt['Charge_Curtailment_MW'],
data_opt['Charge_MW'], data_opt['Discharge_MW'], data_opt['SoC_MWh']
```

Per-hour for the first 720 hours (a 30-day window). Annual revenue is then **extrapolated by ×(8760/720)** (`app.py:138`) — a linear scale-up that assumes the 30-day sample is representative. A real diligence model would use 8,760-hour optimization or seasonal typical weeks.

---

## 4. Technical Mechanics — "How It Works"

### Data flow (6 steps)

1. **User inputs** sidebar (`app.py:24–38`): Power MW, Duration h, RTE %, CAPEX $/kWh, OPEX $/kW-yr, Lifespan, ITC %.
2. **Synthetic data generation** (`generate_energy_data`, `app.py:48–66`): fixed seed (`42`), diurnal sine curve + Gaussian noise for price, 5% probability of negative prices, generation correlated to curtailment.
3. **PuLP LP build + CBC solve** on first 720 hours (`optimize_bess`, lines 84–126).
4. **Extrapolation + CAPEX/OPEX math** (lines 138–147): linear ×12.17 scale, ITC haircut, flat cash-flow annuity.
5. **NPV @ 8% fixed** (line 146) + payback = CAPEX / annual cash flow.
6. **Plotly visualization** (SoC trace, cumulative cash-flow bar chart) and optional **Gemini executive summary** from the headline metrics.

### Layer responsibilities

- **Streamlit** — input capture, session state, result rendering; caches synthetic data with `@st.cache_data`.
- **PuLP + CBC** — solves the 720×4 = 2,880-variable LP in <1 s.
- **Gemini (`gemini-2.5-flash`)** — narrative only. **It does not see dispatch data** — only four scalars.

### The exact Gemini prompt (`app.py:207–211`)

```python
prompt = f"""
Act as an energy strategy consultant. Analyze the following BESS financial metrics and provide a 2-paragraph executive summary on the project viability.
Metrics: Net CAPEX: ${metrics['net_capex']:,.2f}, Annual Cash Flow: ${metrics['annual_cash_flow']:,.2f}, NPV (8% discount): ${metrics['npv']:,.2f}, Payback Period: {metrics['payback_period']:.2f} years.
Context: The system optimizes revenue by charging from zero-cost curtailed renewable energy and discharging during high nodal prices, avoiding negative pricing penalties. Focus strictly on financial and strategic implications. Do not use positive framing. Maintain a factual tone.
"""
```

The "Do not use positive framing" directive is the only real instruction shaping the output — the model is being used as a neutral memo-drafter, not as a reasoning engine.

### External data sources — current state vs. needed

| Data source | In code? | What it should be |
|---|---|---|
| LMP prices | **No** — synthetic `30 + 20·sin + 15·randn` | ISO APIs (CAISO OASIS, ERCOT MIS, PJM DataMiner2, NYISO OASIS) |
| Curtailment data | **No** — derived from price < threshold | EIA-930, ISO curtailment reports, site SCADA |
| eGRID / carbon factors | **No** | EPA eGRID subregion emission rates |
| IRA bonus-adder eligibility | **No** | DOE Energy Community map, Treasury domestic-content guidance |
| Degradation / warranty | **No** | OEM warranty curves (Tesla Megapack, Fluence, Powin) |

### Hardcoded values that must become inputs for production

| Constant | Location | Production default |
|---|---|---|
| Discount rate 8% | `app.py:146` | WACC-driven, project-specific (7–10%) |
| Random seed 42 | `app.py:49` | Remove — use real data |
| Optimization window 720 hours | `app.py:130` | Full 8,760 or 52×168-hour typical weeks |
| Negative-price frequency 5% | `app.py:52` | Node-specific from historic LMPs |
| Curtailment factor thresholds ($10, $0) | `app.py:57` | Operator-specific PPA curtailment clauses |
| ITC as flat % of CAPEX | `app.py:141` | §48E base + bonus adders + basis reduction |
| Gemini model id `gemini-2.5-flash` | `app.py:205` | Config/env-driven |

---

## 5. Consulting Lens — Big-Picture View

This tool sits in **pre-financial-close origination**, specifically at the **go/no-go screening stage** when an IPP development team is deciding whether to co-locate BESS at an existing wind or solar site. It is *not* an operational dispatch tool (too stylized) and *not* a portfolio / IRP planning tool (single-asset scope).

The target persona is the **IPP development analyst or corporate-development associate** who today opens Excel, pulls a curtailment report from the asset-management team, assumes a price shape, and hand-builds a dispatch pro forma — a 1–2 week exercise this tool collapses to 30 seconds. It is not yet useful to infrastructure fund LP / GP underwriting teams (needs capital-stack modeling) nor to utility IRP modelers (needs production-cost integration).

The **co-location thesis is the correct timing bet for 2026**: IRA §48E is fully available to standalone storage (a 2022 change), hybrid PPAs are displacing single-technology PPAs, interconnection queues are gate-locked (making co-location the only near-term path to MWs on the grid), and FERC Orders 841 and 2222 have normalized storage and DER participation in wholesale markets. A tool that quantifies *the specific value unlock of putting a battery behind an existing interconnection* captures exactly that tailwind.

**What becomes multi-dimensional once deployed**: developers begin to underwrite sites for curtailment optionality rather than avoiding curtailed nodes; utilities start stress-testing transmission upgrade deferrals by pricing BESS as an alternative; infra funds reprice the terminal value of existing renewable portfolios upward (because every unsaturated interconnection is now optionality, not a stranded asset).

**Why incumbents haven't built exactly this**: Aurora, PLEXOS, and Energy Exemplar optimize production-cost or capacity-expansion at the system level and charge $100k+/seat — their business model is utility IRP and ISO planning, not single-asset pro formas. HOMER targets microgrids. The gap is a **deal-level analyst tool that speaks both LP dispatch and IRA capital stack in one workflow** — culturally closer to Deloitte's consulting product than to a software vendor's.

---

## 6. Future Roadmap

### Three next features that compound value

1. **Real LMP ingestion + nodal vs. zonal toggle** — replace the synthetic series with CAISO OASIS / ERCOT MIS / PJM DataMiner2 historicals, plus a forward-curve scenario module (flat, Aurora-style mean-reverting, GenX fundamentals). This single change transforms the tool from prototype to diligence-grade.
2. **Stacked-revenue layer** — ancillary services (FRAS in CAISO, FFR in ERCOT, Reg-D in PJM), capacity auction clearing, Resource Adequacy value. Co-optimize under cycle-life constraints so the model trades off arbitrage vs. AS reservation.
3. **IRA capital-stack module** — full §48E base (6% / 30%) + energy-community (+10%) + domestic-content (+10%) + low-income (+10/+20%) adders; §6418 transferability pricing (~92¢ on the dollar market); tax-equity sizing; back-leverage debt sculpting with DSCR ≥ 1.30×.

### Path to "renewable portfolio underwriting platform"

Today: single-asset BESS at one node. Progression:
- (a) multi-technology hybrid (solar + wind + BESS at one POI)
- (b) portfolio roll-up across nodes with correlated price / curtailment scenarios
- (c) Monte Carlo on LMP path + basis risk + curtailment volatility
- (d) capital-stack optimizer for the full portfolio (tax equity + sponsor equity + back-leverage)
- (e) M&A / secondary-market valuation of operating portfolios — where infrastructure funds buy.

### Adjacent problems

**Green hydrogen §45V modeling** (electrolyzer dispatch is the same LP template with a different product price curve and three-pillars accounting); **microgrid / campus optimization**; **EV fleet depot charging** against demand charges + TOU; **long-duration storage** (8–12h Form Energy-style) where cycle life dominates; **virtual power plant aggregation** under FERC Order 2222.

### Regulatory tailwinds

FERC Order 841 (storage in wholesale markets, in effect), FERC Order 2222 (DER aggregation, implementing), FERC Order 1920 (transmission planning), CAISO ELCC for storage, ERCOT real-time co-optimization go-live, state clean-peak standards (MA, CA), and the IRA's §48E runway through 2032+.

### What makes a utility CFO or infra-fund GP sign a 7-figure check

Three must-haves:
1. **Defensible price forecast methodology** — fundamentals-backed, not stochastic-only.
2. **Audited tax-equity / transferability model** — matches what K&L Gates or Skadden would paper.
3. **Integration with the asset-management stack** — SCADA, curtailment reports, settlement data — so it is a live tool, not a memo-stage spreadsheet.

The check lands when the tool demonstrates measurably better IRR on a real closed deal vs. the comparable Excel-based underwriting.

---

## 7. What's Missing — The Honest Gap List

### Documented but not built

- **"Payback periods"** (README) — implemented as `net_capex / annual_cash_flow` (`app.py:147`), undiscounted, ignoring ITC timing.
- **"Annual cash flows"** (README, plural) — the code produces a **flat annuity**: `[-net_capex] + [annual_cash_flow] * lifespan` (`app.py:145`). No year-over-year variation, no escalation, no degradation.

### In code but undocumented

- Extrapolation factor `8760/720` (documented nowhere).
- Fixed random seed 42 — the entire data series is deterministic, so every user sees the same "results."
- The `.github/workflows/keep_alive.yml` + `ping.py` combo exists purely to prevent Streamlit Community Cloud from idling the free-tier app. This is scaffolding, not product.

### Energy-industry credibility gaps

| Topic | Current state | Gap |
|---|---|---|
| Degradation modeling | **Absent** | No capacity fade (typical 2%/yr, then plateau), no resistance growth |
| Warranty integration | **Absent** | No OEM cycle-count or throughput warranty modeling |
| Augmentation strategy | **Absent** | Year-10 augmentation CAPEX ignored |
| Capacity fade | **Absent** | `energy_capacity_mwh` constant over 20 years |
| Cycle-life curves | **Absent** | No cycle or throughput constraint in LP |
| Price forecasting | **Synthetic sine+noise** | No fundamentals, no forward curves, no scenarios |
| LMP nodal vs. zonal | **Neither — pure synthetic** | Real tool must handle basis and congestion |
| Curtailment stress testing | **Absent** | Single deterministic curtailment series |
| DSCR / debt sculpting | **Absent** | 100% equity assumption |
| ITC basis reduction | **Absent** | §48(a)(6) requires CAPEX basis reduced by 50% of ITC |
| IRA bonus adders | **Absent** | Flat % only |
| §6418 transferability | **Absent** | Market discount on credit sales not modeled |
| Interconnection costs | **Absent** | Network upgrades and IX study costs not in CAPEX |
| Property tax / insurance | **Absent** | OPEX is a single $/kW-yr scalar |
| Contract layers | **Absent** | No PPA, tolling, or merchant/hedged split |

### What a Deloitte partner must demand before showing this to an energy client

1. **Replace synthetic data with real ISO LMP historicals** for at least one test node (a CAISO SP-15 or ERCOT West node with visible curtailment history).
2. **Full-year 8,760-hour optimization** — kill the 720-hour extrapolation; it's the single most indefensible shortcut.
3. **Degradation + augmentation schedule** — even a linear 2%/yr fade with a year-10 augmentation line item passes the smell test.
4. **Capital stack module** — tax equity + back-leverage with DSCR constraint, or the model isn't credible to an IPP CFO.
5. **IRA bonus adder logic** keyed to site address (energy-community overlay) — the single most commercially important IRA feature for 2026 deals.
6. **Price-scenario Monte Carlo** — or at minimum three named scenarios (fundamentals low / base / high).
7. **Sensitivity table** on NPV (tornado chart on RTE, CAPEX, price shape, discount rate, degradation).
8. **Remove the hardcoded 8% discount rate** and reframe the Gemini layer — today it adds no quantitative value. Either remove from the pitch, or reposition as an "auto-draft IC memo" feature that consumes the dispatch data, not four summary scalars.

---

## Bottom Line

A **credible proof-of-concept that correctly identifies the commercial thesis** (co-located BESS curtailment capture under IRA), shipped as a **single-asset, synthetic-data, flat-cash-flow prototype**. It is the right skeleton. The consulting narrative should position the Section 6 roadmap as the productization plan — not oversell the current artifact.
