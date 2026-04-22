# BESS Value Architect

Deal-level underwriting for co-located battery storage at renewable sites.
Built for IPP development and infrastructure-fund analysts evaluating whether
to put a 4-hour battery behind an existing wind/solar interconnection.

## What's new

Three-file engine (`app.py` / `engine.py` / `data.py`) running real CAISO
OASIS data through a full-year 8,760-hour LP and a capital-stack model with
tax equity, sculpted back-leverage debt, IRA §48E adders, and — new in this
cut — three switchable augmentation funding methods (Reserve / PAYG /
Debt-refi) compared side-by-side. Debt under the Reserve method is sized
with the sinking-fund contribution included in senior obligations (real-
world DSRA practice), so Min DSCR (senior) hits the 1.30× covenant while
Min DSCR (debt only) floats above it — both shown in the UI. The capital-
structure tradeoff is deliberate: Reserve de-levers the project for risk
reduction and covenant compliance, not IRR maximisation.

## Smoke-test headline numbers

Default inputs (50 MW / 4 hr, $300/kWh, RTE 85%, CAPEX fade 2%/yr, 20-yr
life) with Energy Community adder on, solved on the bundled SP-15 2024
dataset:

| Metric | PAYG | Reserve | Debt-refi |
|---|---|---|---|
| Sponsor IRR | **13.6%** | 9.2% | **16.1%** |
| Project IRR | 13.6% | 13.6% | 13.6% |
| Min DSCR (senior) | 1.30× | 1.30× | 1.30× |
| Min DSCR (debt only) | 1.30× | **2.45×** | 1.30× |
| Sources & Uses | $12.6M TE + $37.2M Debt + $10.2M Sponsor | $12.6M TE + $17.7M Debt + $29.7M Sponsor | $12.6M TE + $37.2M Debt + $10.2M Sponsor |

The three-method comparison is the **feature**, not a flaw. Sponsor IRR
ordering (debt-refi > PAYG > reserve) reflects the direct relationship
between leverage and equity return: reserving cash senior to sponsor
reduces the project's debt capacity and mechanically lowers Sponsor IRR.
Reserve is chosen when lenders require a DSRA covenant or when year-10
equity shock-absorption is uninsurable; otherwise PAYG (or debt-refi, if
incremental leverage is acceptable) dominates.

## What it does

1. **Pulls real CAISO DA LMPs** (SP-15 / NP-15 / ZP-26) from the OASIS public
   API. Caches to Parquet. Falls back to a deterministic bundled series
   calibrated to published 2024 statistics if OASIS is unreachable — the UI
   banner always states which path was taken.
2. **Optimises 8,760-hour dispatch** via a PuLP linear program with
   curtailment-as-zero-cost fuel, a split-efficiency SoC balance
   (η̂ = √η), and a terminal SoC closure to remove the horizon-emptying
   bias of short-window LPs.
3. **Builds an IRA §48E capital stack**: base 30% + optional Energy Community
   (+10%) + Domestic Content (+10%) adders, with §48(a)(6) basis reduction.
   Tax equity sized at ITC × yield multiple. Back-leverage debt sculpted to
   the minimum DSCR target across the tenor. Sponsor equity is the residual.
4. **Models degradation + augmentation**: linear annual fade, year-N
   augmentation CAPEX restoring to a configurable fraction of nameplate.
   Year-over-year revenue scales with capacity ratio.
5. **Outputs Project IRR, Sponsor IRR, TE IRR, Project NPV, Min DSCR,
   discounted payback**, an hourly dispatch chart for a representative week,
   a year-by-year cash-flow waterfall, and DSCR-by-year.
6. **Tornado sensitivity** on Project NPV across RTE, CAPEX, discount rate,
   degradation rate, the Energy Community adder, and price shape
   (hub-vs-node volatility proxy).
7. **Drafts an IC memo** via Gemini 2.5 Flash in the standard format:
   Thesis · Sources of Value · Key Risks · Sensitivities · Recommendation.
   The memo consumes the full analysis output — not a handful of summary
   scalars — and is constrained to cite the model's figures.

## File layout

```
app.py         Streamlit UI
engine.py      LP + project finance + sensitivity
data.py        CAISO OASIS fetcher + Parquet cache + bundled fallback
requirements.txt
```

## Industry-sourced defaults

Every default value traces to a named source and is overridable in the UI.

| Parameter | Default | Source |
|---|---|---|
| ITC base | 30% | IRC §48E(a)(1)(A) |
| Energy Community adder | +10% | IRC §48E(a)(3)(A) — DOE EC Bonus Map |
| Domestic Content adder | +10% | IRC §48E(a)(3)(B) — IRS Notice 2024-41 |
| Basis reduction | 50% of ITC | IRC §48(a)(6) |
| RTE | 85% | Tesla Megapack 2 XL spec; Fluence Gridstack |
| CAPEX | $300/kWh | Lazard LCOS 2024 v9 (4-hr LFP midpoint) |
| Degradation | 2%/yr | NREL 2023 Cost & Performance Projections |
| DSCR target | 1.30× | NREL ATB 2024 back-leverage covenant |
| Debt rate | 7% | 5-yr UST + ~250 bp spread (2024–25) |
| TE yield multiple | 1.10 | 2024 §6418 transferability secondary market |

## Running

```bash
pip install -r requirements.txt
streamlit run app.py
```

Provide `GEMINI_API_KEY` via `st.secrets` or paste it into the sidebar for
the IC memo feature.

## Known simplifications

- Site-level generation and curtailment are not published on OASIS. They are
  proxied from a calibrated 2-axis-tracking solar profile, with curtailment
  re-triggered by the real LMP signal so the arbitrage/curtailment split
  remains physically consistent with the real price series.
- Partnership-flip distribution is simplified to a fixed pre/post-PAYGO
  cash share rather than target-yield-driven flip tracking. Acceptable for
  screening; a full deal model needs yield-based flip.
- MACRS 5-year depreciation tax shields (allocated to TE in real
  partnership flips) are **not** explicitly modelled. TE IRR reported by
  this tool is therefore understated relative to a full tax-equity
  underwriting model. The Project and Sponsor IRRs are not affected by
  this simplification because depreciation is a pass-through item.
- RTE sensitivity in the tornado uses a linear revenue-scaling approximation
  around the base dispatch to avoid re-running a minute-scale LP per
  perturbation.
- Arbitrage and curtailment-capture are the only revenue streams modelled.
  Ancillary services, capacity payments, and resource-adequacy credits are
  not included.
- No cycle-life or throughput constraint on the LP; capacity fade is
  purely calendar-based. In practice, OEM warranties cap cycles/year
  (typically 365–500 for 4-hr LFP), which would reduce the optimised
  cycle count relative to this tool's outputs.
