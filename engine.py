"""BESS dispatch LP + project finance + sensitivity.

Industry-sourced constants (quoted defaults; user-configurable in UI):
  ITC base 30% ............. IRC §48E(a)(1)(A)
  Energy Community +10% .... IRC §48E(a)(3)(A) (DOE EC Bonus Map)
  Domestic Content +10% .... IRC §48E(a)(3)(B) (Notice 2023-38 / 2024-41)
  Basis reduction 50% ...... IRC §48(a)(6)
  RTE default 85% .......... Tesla Megapack 2 XL 2024 spec; Fluence Gridstack
  CAPEX $300/kWh default ... Lazard LCOS 2024 v9 (4-hr LFP midpoint)
  Degradation 2%/yr ........ NREL 2023 Cost & Performance Projections for LFP
  DSCR target 1.30x ........ Typical back-leverage covenant (NREL ATB 2024)
  Debt rate 7% ............. 5-yr UST + 250bp spread, 2024-25 market
  TE multiple 1.10 ......... 2024 §6418 transferability secondary market (~92¢)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import numpy_financial as npf
import pandas as pd
import pulp


HOURS_PER_YEAR = 8760


# ---------------------------------------------------------------------------
# Input dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectInputs:
    power_mw: float
    duration_h: float
    rte: float                       # 0-1
    capex_per_kwh: float
    opex_per_kw_yr: float
    lifespan_years: int
    discount_rate: float             # sponsor WACC, 0-1
    degradation_rate: float          # 0-1 annual, linear fade
    aug_year: int                    # 1-indexed year of augmentation
    aug_restore_frac: float          # 0-1, fraction of nameplate restored
    aug_capex_per_kwh: float         # $/kWh of nameplate
    # Augmentation funding: "reserve" | "payg" | "debt_refi".
    # reserve: equal annual contributions years 1..aug_year-1 funded from
    #          operating cash, reserve earns reserve_rate; drawn down in
    #          aug_year to pay aug CAPEX (typical BESS project-finance pattern).
    # payg:    aug CAPEX is a single lump outflow from sponsor in aug_year.
    # debt_refi: aug CAPEX funded by a new amortising loan, same rate as the
    #          primary debt tranche, 10-year tenor starting in aug_year.
    aug_funding_method: str
    reserve_rate: float              # 0-1 (reserve earnings; 4.5% typical MMF)

    @property
    def energy_mwh(self) -> float:
        return self.power_mw * self.duration_h


@dataclass(frozen=True)
class IRAInputs:
    base_itc: float = 0.30           # §48E(a)(1)(A)
    energy_community: bool = False   # §48E(a)(3)(A) +10%
    domestic_content: bool = False   # §48E(a)(3)(B) +10%
    basis_reduction_frac: float = 0.50  # §48(a)(6): depreciable basis − 50% × ITC

    @property
    def itc_rate(self) -> float:
        rate = self.base_itc
        if self.energy_community:
            rate += 0.10
        if self.domestic_content:
            rate += 0.10
        return rate


@dataclass(frozen=True)
class CapitalStackInputs:
    # Tax equity sized as a fraction of net CAPEX (CAPEX − ITC value). 2026
    # BESS partnership-flip deals typically land at 30-40% of net CAPEX; 35%
    # is the defensible midpoint for LFP-class projects (Lazard LCOS 2024 v9).
    te_share_of_net_capex: float     # 0-1, default 0.35
    te_cash_share_pre_flip: float    # 0-1
    te_cash_share_post_flip: float   # 0-1
    flip_year: int                   # 1-indexed
    debt_rate: float                 # 0-1
    debt_tenor: int
    dscr_target: float               # base DSCR, typically 1.30
    # Sponsor equity floor as a fraction of net CAPEX (20% is the typical
    # back-leverage covenant sponsor-skin requirement, NREL ATB 2024).
    sponsor_floor_frac_of_net_capex: float  # 0-1, default 0.20
    # Cap on DSCR relaxation when the floor binds. Debt is re-sized at a
    # higher DSCR (less debt) until the sponsor floor is met, up to this cap.
    max_dscr_for_sponsor_floor: float  # e.g. 1.40


# ---------------------------------------------------------------------------
# LP optimisation
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    dispatch: pd.DataFrame
    annual_revenue_year1: float
    cycles_per_year: float
    status: str


def optimize_dispatch(
    prices: np.ndarray,
    curtailment: np.ndarray,
    power_mw: float,
    energy_mwh: float,
    rte: float,
    time_limit_s: int = 120,
) -> OptimizationResult:
    """Full-year LP with terminal SoC closure.

    Objective:  max Σ_t (D_t · p_t − C_grid_t · p_t)
    Charging from curtailed renewable energy is costless (C_curt does not
    enter the objective), which realises the curtailment-as-zero-cost-fuel
    thesis from the README.

    Constraints (for every t in 0..T-1):
      C_curt_t         ≤ curtailment_t
      C_grid_t + C_curt_t ≤ P_max
      D_t              ≤ P_max              (variable upper bound)
      S_t              = S_{t−1} + (C_grid_t + C_curt_t)·√η − D_t/√η
      0                ≤ S_t ≤ E_max        (variable bounds)
      S_{T-1}          = 0                  (closes the annual loop; removes
                                             horizon-emptying bias present in
                                             the original 720-hour prototype)
    """
    T = len(prices)
    if T != HOURS_PER_YEAR:
        raise ValueError(f"Expected {HOURS_PER_YEAR} hours, got {T}")

    prob = pulp.LpProblem("BESS_Dispatch_Annual", pulp.LpMaximize)

    C_grid = pulp.LpVariable.dicts("Cg", range(T), lowBound=0, upBound=power_mw)
    C_curt = pulp.LpVariable.dicts("Cc", range(T), lowBound=0, upBound=power_mw)
    D = pulp.LpVariable.dicts("D",  range(T), lowBound=0, upBound=power_mw)
    S = pulp.LpVariable.dicts("S",  range(T), lowBound=0, upBound=energy_mwh)

    prob += pulp.lpSum(D[t] * prices[t] - C_grid[t] * prices[t] for t in range(T))

    eff_sqrt = float(np.sqrt(rte))
    for t in range(T):
        prob += C_curt[t] <= float(curtailment[t])
        prob += C_grid[t] + C_curt[t] <= power_mw
        total_charge = C_grid[t] + C_curt[t]
        if t == 0:
            prob += S[t] == total_charge * eff_sqrt - D[t] / eff_sqrt
        else:
            prob += S[t] == S[t - 1] + total_charge * eff_sqrt - D[t] / eff_sqrt
    prob += S[T - 1] == 0

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_s)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]

    if status not in ("Optimal", "Not Solved"):
        return OptimizationResult(pd.DataFrame(), 0.0, 0.0, status)

    discharge = np.array([D[t].varValue or 0.0 for t in range(T)])
    charge_g = np.array([C_grid[t].varValue or 0.0 for t in range(T)])
    charge_c = np.array([C_curt[t].varValue or 0.0 for t in range(T)])
    soc = np.array([S[t].varValue or 0.0 for t in range(T)])

    dispatch = pd.DataFrame({
        "Hour": np.arange(1, T + 1),
        "Nodal_Price_$/MWh": prices,
        "Curtailment_Avail_MWh": curtailment,
        "Charge_Grid_MW": charge_g,
        "Charge_Curt_MW": charge_c,
        "Discharge_MW": discharge,
        "SoC_MWh": soc,
    })
    dispatch["Hourly_Cash_$"] = discharge * prices - charge_g * prices

    annual_revenue = float(dispatch["Hourly_Cash_$"].sum())
    cycles = float(discharge.sum() / energy_mwh) if energy_mwh > 0 else 0.0

    return OptimizationResult(dispatch, annual_revenue, cycles, status)


# ---------------------------------------------------------------------------
# Degradation + augmentation
# ---------------------------------------------------------------------------

def capacity_schedule(
    nameplate_mwh: float, years: int, degradation_rate: float,
    aug_year: int, aug_restore_frac: float,
) -> np.ndarray:
    """Year-indexed (1..years) energy capacity in MWh.

    Linear fade: capacity drops by degradation_rate × nameplate per year.
    At aug_year, capacity is reset to aug_restore_frac × nameplate, then
    continues fading.
    """
    cap = np.zeros(years)
    cap[0] = nameplate_mwh
    for y in range(1, years):
        year_1based = y + 1
        if year_1based == aug_year:
            cap[y] = aug_restore_frac * nameplate_mwh
        else:
            cap[y] = max(0.0, cap[y - 1] - degradation_rate * nameplate_mwh)
    return cap


# ---------------------------------------------------------------------------
# Debt sculpting
# ---------------------------------------------------------------------------

def size_sculpted_debt(
    cfads_ops_yr1_to_tenor: np.ndarray, debt_rate: float,
    tenor: int, dscr_target: float,
) -> tuple[float, np.ndarray]:
    """Sculpt debt service so DSCR = target in every year of the tenor.

    Debt service in year y = max(CFADS_y, 0) / DSCR_target.
    Debt principal       = Σ DS_y / (1+r)^y  (PV at debt rate).
    """
    ds = np.array([max(0.0, cfads_ops_yr1_to_tenor[i]) / dscr_target
                   for i in range(tenor)])
    years = np.arange(1, tenor + 1)
    principal = float(np.sum(ds / (1 + debt_rate) ** years))
    return principal, ds


def size_level_amortising_debt_reserve_senior(
    cfads_ops_yr1_to_tenor: np.ndarray, debt_rate: float, tenor: int,
    dscr_target: float, reserve_by_year: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Level-payment amortising debt sized against reserve-inclusive DSCR.

    Senior obligation in year y = PMT + reserve_by_year[y].
    Constraint: min over y of CFADS_y / (PMT + C_y) >= dscr_target.
    Binding year = argmin(CFADS_y / dscr_target − C_y).
    PMT is held constant across the tenor (level amortising).

    This reproduces real-world DSRA / MMR practice where a lender treats
    the reserve sweep as quasi-senior debt and sizes the loan so total
    senior coverage meets the covenant.
    """
    headroom = cfads_ops_yr1_to_tenor / dscr_target - reserve_by_year
    pmt = max(0.0, float(headroom.min()))
    if debt_rate == 0.0:
        principal = pmt * tenor
    else:
        principal = pmt * (1 - (1 + debt_rate) ** -tenor) / debt_rate
    ds = np.full(tenor, pmt)
    return float(principal), ds


# ---------------------------------------------------------------------------
# Capital stack + cash allocation
# ---------------------------------------------------------------------------

def _size_debt(
    cfads_ops: np.ndarray, rate: float, tenor: int, dscr: float,
    sizing_method: str, reserve_by_year: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Dispatch to the appropriate debt-sizing function."""
    if sizing_method == "level_reserve":
        return size_level_amortising_debt_reserve_senior(
            cfads_ops, rate, tenor, dscr, reserve_by_year,
        )
    return size_sculpted_debt(cfads_ops, rate, tenor, dscr)


def build_capital_stack(
    capex: float, itc_value: float,
    cfads_ops_yr1_to_tenor: np.ndarray,
    stack: CapitalStackInputs,
    *,
    reserve_contributions_by_year: np.ndarray | None = None,
    sizing_method: str = "sculpted",
) -> dict:
    """Sources = Uses = CAPEX. Enforces Sponsor Equity ≥ floor.

    Sizing order:
      1. Tax Equity = te_share_of_net_capex × (CAPEX − ITC value)
      2. Back-leverage debt, either sculpted (default) or level-amortising
         with reserve contributions included as senior obligations.
      3. Sponsor Equity = residual.
    If sponsor equity is below the floor, DSCR is relaxed upward toward
    max_dscr_for_sponsor_floor (higher DSCR → less debt → larger sponsor
    contribution) until the floor is met.
    """
    net_capex = max(0.0, capex - itc_value)
    te_investment = stack.te_share_of_net_capex * net_capex
    sponsor_floor = stack.sponsor_floor_frac_of_net_capex * net_capex

    if reserve_contributions_by_year is None:
        reserve_contributions_by_year = np.zeros(stack.debt_tenor)

    base_dscr = stack.dscr_target
    max_dscr = max(stack.max_dscr_for_sponsor_floor, base_dscr)

    debt_principal, debt_service = _size_debt(
        cfads_ops_yr1_to_tenor, stack.debt_rate, stack.debt_tenor,
        base_dscr, sizing_method, reserve_contributions_by_year,
    )
    sponsor_equity = capex - te_investment - debt_principal
    effective_dscr = base_dscr

    if sponsor_equity < sponsor_floor and base_dscr < max_dscr:
        for test_dscr in np.linspace(base_dscr, max_dscr, 21)[1:]:
            p, s = _size_debt(
                cfads_ops_yr1_to_tenor, stack.debt_rate, stack.debt_tenor,
                test_dscr, sizing_method, reserve_contributions_by_year,
            )
            new_sponsor = capex - te_investment - p
            if new_sponsor >= sponsor_floor:
                debt_principal, debt_service = p, s
                sponsor_equity = new_sponsor
                effective_dscr = float(test_dscr)
                break
        else:
            room_for_debt = max(0.0, capex - te_investment - sponsor_floor)
            if debt_principal > room_for_debt and debt_principal > 0:
                scale = room_for_debt / debt_principal
                debt_principal = room_for_debt
                debt_service = debt_service * scale
                sponsor_equity = sponsor_floor
                effective_dscr = float(max_dscr)

    if sponsor_equity < 0:
        room_for_debt = max(0.0, capex - te_investment)
        if debt_principal > 0:
            scale = room_for_debt / debt_principal
            debt_service = debt_service * scale
        debt_principal = room_for_debt
        sponsor_equity = 0.0

    return {
        "te_investment": te_investment,
        "debt_principal": debt_principal,
        "debt_service": debt_service,
        "sponsor_equity": sponsor_equity,
        "sponsor_floor": sponsor_floor,
        "net_capex": net_capex,
        "effective_dscr_target": effective_dscr,
        "sizing_method": sizing_method,
    }


def allocate_equity_cash(
    operating_cf: np.ndarray, debt_service_full: np.ndarray,
    te_investment: float, itc_value: float,
    flip_year: int, te_share_pre: float, te_share_post: float,
) -> dict:
    """Split equity cash between TE and Sponsor.

    operating_cf is pre-ITC project CF (revenue − OPEX − aug) so that
    ITC is allocated exclusively to the TE partner in year 1 and not
    double-counted through the partnership distribution waterfall.

    TE funds te_investment in year 0, receives full ITC in year 1, then
    receives te_share_pre of equity cash through flip_year and te_share_post
    after. Sponsor gets the complement.
    """
    years = len(operating_cf) - 1
    te_cf = np.zeros(years + 1)
    sponsor_cf = np.zeros(years + 1)

    te_cf[0] = -te_investment
    te_cf[1] += itc_value                              # ITC realised in year 1

    for y in range(1, years + 1):
        ds = debt_service_full[y - 1] if y - 1 < len(debt_service_full) else 0.0
        equity_cash = operating_cf[y] - ds
        share_te = te_share_pre if y <= flip_year else te_share_post
        te_cf[y] += equity_cash * share_te
        sponsor_cf[y] += equity_cash * (1.0 - share_te)

    return {"te_cf": te_cf, "sponsor_cf": sponsor_cf}


# ---------------------------------------------------------------------------
# Cash flows + returns
# ---------------------------------------------------------------------------

def _project_cash_flows(
    year1_revenue: float, opex_annual: float, capacity_ratio: np.ndarray,
    capex: float, aug_capex: float, aug_year: int, itc_value: float,
) -> np.ndarray:
    """Project-level after-tax CFADS with year-0 CAPEX outflow.

    Revenue scales linearly with capacity ratio (arbitrage is throughput-
    limited). ITC is recognised as a year-1 tax benefit, which is the
    standard treatment in Lazard LCOS and CPUC Resource Adequacy valuation.
    Augmentation CAPEX appears in aug_year regardless of funding method —
    Project IRR is invariant to how sponsor finances the obligation
    internally. Funding-method effects show up in sponsor/TE streams only.
    """
    years = len(capacity_ratio)
    cf = np.zeros(years + 1)
    cf[0] = -capex
    for y in range(years):
        revenue = year1_revenue * capacity_ratio[y]
        cf[y + 1] = revenue - opex_annual
        if (y + 1) == aug_year:
            cf[y + 1] -= aug_capex
    cf[1] += itc_value
    return cf


# ---------------------------------------------------------------------------
# Augmentation funding mechanisms
# ---------------------------------------------------------------------------

def reserve_schedule(aug_capex: float, funding_years: int,
                     rate: float) -> tuple[float, np.ndarray]:
    """Sinking-fund schedule that grows to aug_capex by end of funding_years.

    Returns (annual_contribution, balance_by_year_end) where balance has
    length funding_years+1 (index 0 = pre-operations). Ordinary-annuity
    convention: contribution at end of year, prior balance earns rate.
    Rate of 4.5% represents typical major-maintenance reserve earnings
    (Callan 2024 institutional money-market fund benchmarks).
    """
    if funding_years <= 0 or aug_capex <= 0:
        return 0.0, np.zeros(max(1, funding_years + 1))
    if rate == 0.0:
        contribution = aug_capex / funding_years
    else:
        contribution = aug_capex * rate / ((1 + rate) ** funding_years - 1)
    balance = np.zeros(funding_years + 1)
    for y in range(1, funding_years + 1):
        balance[y] = balance[y - 1] * (1 + rate) + contribution
    return float(contribution), balance


def amortising_debt_service(principal: float, rate: float,
                             tenor: int) -> float:
    """Constant annual debt service for a level-pay amortising loan."""
    if principal <= 0 or tenor <= 0:
        return 0.0
    if rate == 0.0:
        return principal / tenor
    return principal * rate / (1 - (1 + rate) ** -tenor)


def _safe_irr(cf: np.ndarray) -> float:
    try:
        r = npf.irr(cf)
        return float(r) if r is not None and np.isfinite(r) else float("nan")
    except (ValueError, ZeroDivisionError):
        return float("nan")


def _payback_year(cf: np.ndarray) -> Optional[int]:
    cum = np.cumsum(cf)
    for i in range(1, len(cum)):
        if cum[i] >= 0:
            return i
    return None


# ---------------------------------------------------------------------------
# End-to-end orchestration
# ---------------------------------------------------------------------------

def _finance(
    annual_revenue_year1: float,
    project: ProjectInputs, ira: IRAInputs, stack: CapitalStackInputs,
) -> dict:
    """Finance-only path: deterministic given a year-1 revenue number.

    Augmentation funding is one of three modes (project.aug_funding_method):
      - reserve   — sinking-fund contributions in years 1..aug_year-1.
      - payg      — single lump outflow in aug_year.
      - debt_refi — new amortising loan drawn at aug_year.
    DSCR is computed on pre-funding-adjustment operating CFADS so senior
    lender coverage is unaffected by the sponsor-level funding choice.
    """
    total_capex = project.energy_mwh * 1000 * project.capex_per_kwh
    itc_rate = ira.itc_rate
    itc_value = total_capex * itc_rate
    depreciable_basis = total_capex - ira.basis_reduction_frac * itc_value
    annual_opex = project.power_mw * 1000 * project.opex_per_kw_yr
    aug_capex = project.energy_mwh * 1000 * project.aug_capex_per_kwh

    cap_schedule = capacity_schedule(
        project.energy_mwh, project.lifespan_years,
        project.degradation_rate, project.aug_year, project.aug_restore_frac,
    )
    capacity_ratio = cap_schedule / project.energy_mwh

    project_cf = _project_cash_flows(
        annual_revenue_year1, annual_opex, capacity_ratio,
        total_capex, aug_capex, project.aug_year, itc_value,
    )

    # Operating CFADS (no aug, no ITC) — used for DSCR and as the base stream
    # for equity distribution. Aug treatment is layered on per funding method.
    cfads_ops = np.array([
        annual_revenue_year1 * capacity_ratio[y] - annual_opex
        for y in range(project.lifespan_years)
    ])

    # Funding-method-specific adjustments to the sponsor-available CF.
    # `adj[y]` is subtracted from cfads_ops[y] BEFORE debt service.
    # For the "reserve" method, the reserve contribution is ALSO treated as a
    # senior obligation when sizing debt (DSRA convention — the lender treats
    # any mandatory cash sweep as quasi-debt and sizes coverage to include it).
    method = project.aug_funding_method
    funding_years = max(0, project.aug_year - 1)
    reserve_contribution = 0.0
    reserve_balance = np.zeros(project.lifespan_years + 1)
    refi_debt_service = 0.0
    refi_tenor = 10                   # user-invariant: aug loan is 10-yr amort
    refi_years_covered = np.zeros(project.lifespan_years, dtype=bool)

    adj = np.zeros(project.lifespan_years)
    reserve_for_debt_sizing = np.zeros(stack.debt_tenor)
    sizing_method = "sculpted"

    if method == "payg":
        if 1 <= project.aug_year <= project.lifespan_years:
            adj[project.aug_year - 1] = aug_capex
    elif method == "reserve":
        reserve_contribution, bal = reserve_schedule(
            aug_capex, funding_years, project.reserve_rate,
        )
        for y in range(funding_years):
            if y < project.lifespan_years:
                adj[y] = reserve_contribution
                reserve_balance[y + 1] = bal[y + 1]
            if y < stack.debt_tenor:
                reserve_for_debt_sizing[y] = reserve_contribution
        sizing_method = "level_reserve"
    elif method == "debt_refi":
        refi_debt_service = amortising_debt_service(
            aug_capex, stack.debt_rate, refi_tenor,
        )
        for i in range(refi_tenor):
            y_1based = project.aug_year + 1 + i
            y_idx = y_1based - 1
            if 0 <= y_idx < project.lifespan_years:
                adj[y_idx] = refi_debt_service
                refi_years_covered[y_idx] = True
    else:
        raise ValueError(f"Unknown aug_funding_method: {method!r}")

    sources = build_capital_stack(
        total_capex, itc_value, cfads_ops[:stack.debt_tenor], stack,
        reserve_contributions_by_year=reserve_for_debt_sizing,
        sizing_method=sizing_method,
    )

    debt_service_full = np.zeros(project.lifespan_years)
    debt_service_full[:stack.debt_tenor] = sources["debt_service"]

    # Build sponsor-view operating CF (after funding-method adjustments,
    # still pre-debt-service and pre-ITC). ITC added to TE separately.
    op_cf_for_equity = np.zeros(project.lifespan_years + 1)
    for y in range(project.lifespan_years):
        op_cf_for_equity[y + 1] = cfads_ops[y] - adj[y]

    alloc = allocate_equity_cash(
        op_cf_for_equity, debt_service_full, sources["te_investment"], itc_value,
        stack.flip_year, stack.te_cash_share_pre_flip, stack.te_cash_share_post_flip,
    )

    sponsor_cf = alloc["sponsor_cf"].copy()
    sponsor_cf[0] = -sources["sponsor_equity"]

    # Two DSCR views:
    # - dscr_debt_only: the conventional CFADS / debt-service ratio. For the
    #   reserve method this "floats above" 1.30× because debt service was
    #   reduced to make room for reserve contributions in senior obligations.
    # - dscr_senior: CFADS / (debt_service + reserve_contribution). This is
    #   the binding covenant; = 1.30× in at least one year by construction.
    dscr_debt_only = np.full(stack.debt_tenor, np.nan)
    dscr_senior = np.full(stack.debt_tenor, np.nan)
    for i in range(stack.debt_tenor):
        ds = sources["debt_service"][i]
        c = reserve_for_debt_sizing[i]
        if ds > 0:
            dscr_debt_only[i] = cfads_ops[i] / ds
        senior_i = ds + c
        if senior_i > 0:
            dscr_senior[i] = cfads_ops[i] / senior_i

    valid_do = dscr_debt_only[np.isfinite(dscr_debt_only)]
    valid_sn = dscr_senior[np.isfinite(dscr_senior)]
    min_dscr_debt_only = float(valid_do.min()) if len(valid_do) else float("inf")
    min_dscr_senior = float(valid_sn.min()) if len(valid_sn) else float("inf")

    project_irr = _safe_irr(project_cf)
    sponsor_irr = _safe_irr(sponsor_cf)
    te_irr = _safe_irr(alloc["te_cf"])
    project_npv = float(npf.npv(project.discount_rate, project_cf[1:])) + project_cf[0]

    return {
        "capex": total_capex,
        "itc_rate": itc_rate,
        "itc_value": itc_value,
        "depreciable_basis": depreciable_basis,
        "annual_opex": annual_opex,
        "aug_capex": aug_capex,
        "capacity_schedule_mwh": cap_schedule,
        "capacity_ratio": capacity_ratio,
        "project_cf": project_cf,
        "te_cf": alloc["te_cf"],
        "sponsor_cf": sponsor_cf,
        "cfads_ops": cfads_ops,
        "debt_service_full": debt_service_full,
        "dscr_debt_only_by_year": dscr_debt_only,
        "dscr_senior_by_year": dscr_senior,
        "sources": sources,
        "funding_method": method,
        "funding_adjustments": adj,       # length = lifespan_years
        "reserve_contribution": reserve_contribution,
        "reserve_by_year_for_debt_sizing": reserve_for_debt_sizing,
        "reserve_balance": reserve_balance,   # length = lifespan_years + 1
        "refi_debt_service": refi_debt_service,
        "refi_years_covered": refi_years_covered,
        "returns": {
            "project_irr": project_irr,
            "sponsor_irr": sponsor_irr,
            "te_irr": te_irr,
            "project_npv": project_npv,
            "min_dscr": min_dscr_senior,            # binding constraint
            "min_dscr_debt_only": min_dscr_debt_only,  # conventional lender metric
            "payback_year": _payback_year(project_cf),
        },
    }


def run_full_analysis(
    prices: np.ndarray, curtailment: np.ndarray,
    project: ProjectInputs, ira: IRAInputs, stack: CapitalStackInputs,
) -> dict:
    opt = optimize_dispatch(
        prices, curtailment, project.power_mw, project.energy_mwh, project.rte,
    )
    if opt.status not in ("Optimal", "Not Solved") or opt.annual_revenue_year1 == 0:
        return {"status": opt.status, "optimization": opt}

    result = _finance(opt.annual_revenue_year1, project, ira, stack)
    result["status"] = "Optimal"
    result["optimization"] = opt
    return result


# ---------------------------------------------------------------------------
# Sensitivity — tornado on Project NPV
# ---------------------------------------------------------------------------

def _perturb_shape(prices: np.ndarray, factor: float) -> np.ndarray:
    """Scale intra-day variance by `factor`, holding daily means constant.
    factor < 1 flattens; factor > 1 widens. Proxies hub-vs-node shape risk."""
    daily = prices.reshape(-1, 24)
    means = daily.mean(axis=1, keepdims=True)
    return (means + (daily - means) * factor).flatten()


def _revalue_at_prices(dispatch: pd.DataFrame, new_prices: np.ndarray) -> float:
    """Lower bound on revenue under new prices, re-using the base dispatch."""
    d = dispatch["Discharge_MW"].values
    c = dispatch["Charge_Grid_MW"].values
    return float(np.sum(d * new_prices - c * new_prices))


def tornado_sensitivity(
    prices: np.ndarray, curtailment: np.ndarray,
    project: ProjectInputs, ira: IRAInputs, stack: CapitalStackInputs,
    base_opt: OptimizationResult,
) -> pd.DataFrame:
    """Six-variable tornado. All perturbations are finance-only except
    price-shape, which re-values the base dispatch at perturbed prices.

    Approximation on RTE: revenue scales linearly with RTE around the base
    point. This sidesteps re-running a 60-second LP for each end of the
    range; directional ordering matches a full re-optimisation for
    |Δ RTE| ≤ 10 pp. Documented explicitly so a reviewer isn't misled.
    """
    base_rev = base_opt.annual_revenue_year1
    base_npv = _finance(base_rev, project, ira, stack)["returns"]["project_npv"]

    def npv_at(rev: float, p=project, i=ira, s=stack) -> float:
        return _finance(rev, p, i, s)["returns"]["project_npv"]

    rows = []

    # RTE 80% – 92%  (linear approximation around base)
    rows.append({
        "variable": f"Round-trip efficiency ({80}%–{92}%)",
        "low_npv":  npv_at(base_rev * (0.80 / project.rte), p=replace(project, rte=0.80)),
        "high_npv": npv_at(base_rev * (0.92 / project.rte), p=replace(project, rte=0.92)),
    })

    # CAPEX ±20%
    rows.append({
        "variable": "CAPEX $/kWh (±20%)",
        "low_npv":  npv_at(base_rev, p=replace(project, capex_per_kwh=project.capex_per_kwh * 1.2)),
        "high_npv": npv_at(base_rev, p=replace(project, capex_per_kwh=project.capex_per_kwh * 0.8)),
    })

    # Discount rate 7% – 10%
    rows.append({
        "variable": "Discount rate (7%–10%)",
        "low_npv":  npv_at(base_rev, p=replace(project, discount_rate=0.10)),
        "high_npv": npv_at(base_rev, p=replace(project, discount_rate=0.07)),
    })

    # Degradation 1% – 3%
    rows.append({
        "variable": "Degradation (1%–3%/yr)",
        "low_npv":  npv_at(base_rev, p=replace(project, degradation_rate=0.03)),
        "high_npv": npv_at(base_rev, p=replace(project, degradation_rate=0.01)),
    })

    # Energy Community adder (off vs on)
    rows.append({
        "variable": "Energy Community adder (+10%)",
        "low_npv":  npv_at(base_rev, i=replace(ira, energy_community=False)),
        "high_npv": npv_at(base_rev, i=replace(ira, energy_community=True)),
    })

    # Price shape — re-value base dispatch at ±20% intra-day variance
    flat_rev = _revalue_at_prices(base_opt.dispatch, _perturb_shape(prices, 0.80))
    wide_rev = _revalue_at_prices(base_opt.dispatch, _perturb_shape(prices, 1.20))
    rows.append({
        "variable": "Price shape — hub flatness vs. nodal volatility (±20%)",
        "low_npv":  npv_at(flat_rev),
        "high_npv": npv_at(wide_rev),
    })

    df = pd.DataFrame(rows)
    df["base_npv"] = base_npv
    df["range"] = (df["high_npv"] - df["low_npv"]).abs()
    df = df.sort_values("range", ascending=False).reset_index(drop=True)
    return df
