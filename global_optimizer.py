"""Global Rebalance optimizer.

Approach
---------
1) Apply hard filters deterministically: sell bonds that outside of limits
2) Solve a single Linear/Integer Programming where the amount of bond being sold goes to cash, so TOTAL portfolio MV is
   fixed.

Objective
---------
Minimize total selling amount:  min(sum_i s_i)

Key modeling choice
-------------------
Because sold MV becomes cash, the denominator for weights is the FIXED total
portfolio market value (including cash). This makes all bucket constraints
linear without the "changing total MV" complication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import re

from base import _standardize_name
from consts import RATING_TO_SCORE


def _extract_rating_label(group: str) -> Optional[str]:
    """
    Extract leading rating token from strings like 'AA+ or worse', 'A- OR WORSE',
    'BBB', 'AAA or better'. Returns normalized uppercase label (e.g., 'AA+').
    """
    s = str(group).strip().upper()
    # take text before 'OR WORSE' / 'OR BETTER' if present
    s = re.split(r"\bOR\s+(WORSE|BETTER)\b", s, maxsplit=1)[0].strip()
    # match typical credit rating tokens: e.g., AAA, AA, A, BBB, BB, B, A+, A-, BBB+, BBB-
    m = re.match(r"^(AAA|AA|A|BBB|BB|B|CCC|CC|C|D)([+-])?$", s)
    if m:
        base, pm = m.groups()
        return base + (pm or "")
    # fallback: first token up to whitespace
    return s.split()[0]


def _as_float(x: Any) -> Optional[float]:
    if x is None or pd.isna(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _rating_threshold_from_df(df: pd.DataFrame, rating_str: str) -> Optional[float]:
    """Map a rating label (e.g., 'AA-') to a numeric score using the portfolio's
    own mapping if available.

    Assumes df contains columns ['rating', 'rating_score'].
    """
    r = str(rating_str).strip().upper()
    mapping = (
        df.dropna(subset=["rating", "rating_score"])
        .groupby(df["rating"].astype(str).str.upper())["rating_score"]
        .median()
        .to_dict()
    )
    return float(mapping.get(r, None)) if r in mapping else None


def _parse_rating_band(group_str: str) -> Tuple[str, str]:
    """Return (hi, lo) rating strings from 'A+ to A-' style input."""
    if not group_str or pd.isna(group_str):
        return ("", "")
    s = str(group_str).strip().upper()
    if "TO" not in s:
        return (s, s)
    parts = [p.strip() for p in s.split("TO")]
    if len(parts) != 2:
        return (s, s)
    return parts[0], parts[1]


def _select_port_mv_band_rule(rules: List[Dict[str, Any]], mv_tot: float) -> Optional[Dict[str, Any]]:
    """Pick the row for 'Max Any Obligor by Port Market Value' based on portfolio size.

    The rules table bands appear to be expressed in $MM (e.g., '<250', '250 - 500', '10,000+').
    So we compare (mv_tot / 1e6) to those ranges.
    """
    mv_mm = mv_tot / 1e6

    def parse_band(band: str) -> Tuple[float, float]:
        b = str(band).strip().replace(",", "")
        if b.endswith("+"):
            lo = float(b[:-1])
            return lo, float("inf")
        if b.startswith("<"):
            hi = float(b[1:])
            return 0.0, hi
        if "-" in b:
            lo_s, hi_s = [x.strip() for x in b.split("-")]
            return float(lo_s), float(hi_s)
        # fallback exact
        v = float(b)
        return v, v

    for r in rules:
        band = r.get("group")
        if band is None or pd.isna(band):
            continue
        lo, hi = parse_band(band)
        if lo <= mv_mm < hi:
            return r
    # if nothing matches, fall back to the most conservative (smallest max_fail)
    best = None
    best_cap = None
    for r in rules:
        cap = _as_float(r.get("max_fail"))
        if cap is None:
            continue
        if best is None or cap < best_cap:
            best, best_cap = r, cap
    return best


@dataclass
class GlobalSolveResult:
    updated_portfolio: pd.DataFrame
    executed_trades: pd.DataFrame
    warning_trades: pd.DataFrame
    target_trades: pd.DataFrame
    constraint_report_pre: pd.DataFrame
    constraint_report_post: pd.DataFrame


class GlobalRebalanceEngine:
    """Run hard filters + global LP constraints in one shot."""

    def __init__(
        self,
        strategy_name: str,
        rule_config: Dict[str, Any],
        min_trade_mv: float = 0.0,
        forbidden_sectors: Optional[Iterable[str]] = None,
        use_integer: bool = True,
    ):
        self.strategy_name = _standardize_name(strategy_name)
        self.rule_config = rule_config
        self.min_trade_mv = float(min_trade_mv)
        self.forbidden_sectors = set([str(x).strip() for x in (forbidden_sectors or [])])
        self.use_integer = use_integer

    # config helpers 
    def _rules(self, section: str) -> List[Dict[str, Any]]:
        cfg = self.rule_config.get(self.strategy_name, {})
        return list(cfg.get(_standardize_name(section), []) or [])

    # hard filters
    def _hard_sells(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Return list of full-sell trades (deterministic)."""
        sells: List[Dict[str, Any]] = []

        # 1) Ratings lower than A+  (Min Security Rating)
        rating_rules = [r for r in self._rules("ratings") if str(r.get("rule")) == "Min Security Rating"]
        if rating_rules and "rating_score" in df.columns:
            min_rating = rating_rules[0].get("min_fail")
            thr = RATING_TO_SCORE[str(min_rating).strip().upper()]
            if thr is not None:
                mask = (~df.get("is_cash", False)) & (df["rating_score"].astype(float) > thr)
                for _, row in df.loc[mask].iterrows():
                    sells.append({
                        "cusip": row.get("cusip"),
                        "obligor": row.get("obligor"),
                        "sell_mv": float(row.get("market_value", 0.0)),
                        "reason": f"Hard filter: rating {row.get('rating')} below A+",
                        "old_rating": row.get("rating"),
                        "weight": row.get("weight"),
                        "rule": "Hard: Min Security Rating",
                    })

        # 2) Maturity: any other year (outside ladder years)
        maturity_rules = [r for r in self._rules("maturity") if str(r.get("rule")) == "Maturity Structure by Market Value"]
        ladder_years: set[int] = set()
        for r in maturity_rules:
            g = r.get("group")
            if g is None or pd.isna(g):
                continue
            gs = str(g).strip()
            if gs.lower().startswith("any other"):
                continue
            try:
                ladder_years.add(int(gs))
            except ValueError:
                continue
        if ladder_years and "maturity_year" in df.columns:
            mask = (~df.get("is_cash", False)) & (~df["maturity_year"].isin(sorted(ladder_years)))
            for _, row in df.loc[mask].iterrows():
                sells.append({
                    "cusip": row.get("cusip"),
                    "obligor": row.get("obligor"),
                    "sell_mv": float(row.get("market_value", 0.0)),
                    "reason": f"Hard filter: maturity {row.get('maturity_year')} outside ladder",
                    "old_rating": row.get("rating"),
                    "weight": row.get("weight"),
                    "rule": "Hard: Outside Ladder",
                })

        # 3) Coupon < 4% (if coupon column exists)
        coupon_cols = [c for c in ["coupon", "coupon_rate"] if c in df.columns]
        if coupon_cols:
            ccol = coupon_cols[0]
            mask = (~df.get("is_cash", False)) & (pd.to_numeric(df[ccol], errors="coerce") < 4.0)
            for _, row in df.loc[mask].iterrows():
                sells.append({
                    "cusip": row.get("cusip"),
                    "obligor": row.get("obligor"),
                    "sell_mv": float(row.get("market_value", 0.0)),
                    "reason": f"Hard filter: coupon {row.get(ccol)} < 4%",
                    "old_rating": row.get("rating"),
                    "weight": row.get("weight"),
                    "rule": "Hard: Coupon < 4%",
                })

        # 4) Forbidden sectors (optional user-provided list)
        if self.forbidden_sectors and "holdings_sector" in df.columns:
            mask = (~df.get("is_cash", False)) & (df["holdings_sector"].astype(str).isin(self.forbidden_sectors))
            for _, row in df.loc[mask].iterrows():
                sells.append({
                    "cusip": row.get("cusip"),
                    "obligor": row.get("obligor"),
                    "sell_mv": float(row.get("market_value", 0.0)),
                    "reason": f"Hard filter: forbidden sector {row.get('holdings_sector')}",
                    "old_rating": row.get("rating"),
                    "weight": row.get("weight"),
                    "rule": "Hard: Forbidden Sector",
                })

        # 5) Taxable / AMT (only if such flags exist in df)
        # Your sample `Bonds.xlsx` doesn't contain these columns, so this is a
        # best-effort hook for your real portfolio schema.
        taxable_cols = ["fed_taxable"]
        if taxable_cols:
            tax_mask = df[taxable_cols].apply(lambda s: s.astype(str).str.lower() == "yes").any(axis=1)
            mask = ~df.get("is_cash", False) & tax_mask

            for _, row in df.loc[mask].iterrows():
                sells.append({
                    "cusip": row.get("cusip"),
                    "obligor": row.get("obligor"),
                    "sell_mv": float(row.get("market_value", 0.0)),
                    "reason": "Hard filter: taxable/AMT bond",
                    "old_rating": row.get("rating"),
                    "weight": row.get("weight"),
                    "rule": "Hard: Taxable/AMT",
                })

        # 6) State hard sell: if bond (not cash) and state is not CA or US
        # (Because we have cash, we only hard-sell bonds.)
        # if "state" in df.columns:
        #     st = df["state"].astype(str).str.strip().str.upper()
        #     # sell if state is missing/blank OR not in allowed set
        #     mask = (~df.get("is_cash", False)) & (~st.isin({"CA", "US"}))
        #     for _, row in df.loc[mask].iterrows():
        #         sells.append({
        #             "cusip": row.get("cusip"),
        #             "obligor": row.get("obligor"),
        #             "sell_mv": float(row.get("market_value", 0.0)),
        #             "reason": f"Hard filter: state {row.get('state')} not in {{CA, US}}",
        #             "old_rating": row.get("rating"),
        #             "weight": row.get("weight"),
        #             "rule": "Hard: State not CA/US",
        #         })

        # de-duplicate by (cusip, obligor) and cap at full MV
        if not sells:
            return []

        tdf = pd.DataFrame(sells)

        # total MV per position (for capping)
        mv_map = (
            df[["cusip", "obligor", "market_value"]]
            .drop_duplicates(subset=["cusip", "obligor"], keep="first")
        )

        # aggregate: sum sell_mv, and CONCAT reasons/rules (keep detail)
        agg = (
            tdf.groupby(["cusip", "obligor"], dropna=False)
            .agg(
                reason=("reason", lambda s: " | ".join(pd.unique(s.dropna().astype(str)))),
                rule=("rule", lambda s: " | ".join(pd.unique(s.dropna().astype(str)))),
                old_rating=("old_rating", "first"),
                weight=("weight", "first"),
            )
            .reset_index()
        )

        out = agg.merge(mv_map, on=["cusip", "obligor"], how="left")
        out["sell_mv"] = out[ "market_value"]
        return out.drop(columns=["market_value"]).to_dict(orient="records")


    # LP constraints 
    def _build_constraints_spec(self, df: pd.DataFrame, mv_tot: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return (constraints, warnings) where constraints are hard LP constraints
        and warnings are informational items.
        """
        constraints: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []

        # # cash cap from Cash % rule (max_fail), for now we don't need it
        # cash_rules = [r for r in self._rules("cash") if str(r.get("rule")) == "Cash %"]
        # if cash_rules:
        #     cash_cap = _as_float(cash_rules[0].get("max_fail"))
        #     if cash_cap is not None and "is_cash" in df.columns:
        #         cash0 = float(df.loc[df["is_cash"].fillna(False), "market_value"].sum())
        #         max_extra_cash = cash_cap * mv_tot - cash0
        #         constraints.append({
        #             "type": "total_sell_cap",
        #             "cap": max_extra_cash,
        #             "name": f"cash_cap_{cash_cap}",
        #         })

        # maturity max_fail per year
        maturity_rules = [r for r in self._rules("maturity") if str(r.get("rule")) == "Maturity Structure by Market Value"]
        for r in maturity_rules:
            g = r.get("group")
            if g is None or pd.isna(g):
                continue
            gs = str(g).strip()
            if gs.lower().startswith("any other"):
                continue # handled by hard filter
            try:
                year = int(gs)
            except ValueError:
                continue
            max_fail = _as_float(r.get("max_fail"))
            if max_fail is not None:
                mask = (~df.get("is_cash", False)) & (df["maturity_year"] == year)
                constraints.append({
                    "type": "bucket_max",
                    "name": f"maturity_{year}_max",
                    "mask": mask,
                    "alpha": max_fail,
                })
            target = _as_float(r.get("target"))
            if target is not None:
                # record as soft target info for reporting
                warnings.append({"rule": "Maturity Target", "group": year, "target": target})

        # rating structure
        rating_rules = [r for r in self._rules("ratings") if str(r.get("rule")) == "Rating Structure"]
        for r in rating_rules:
            group = r.get("group")
            if group is None or pd.isna(group):
                continue
            hi, lo = _parse_rating_band(str(group))
            # We'll build band mask using rating_score thresholds from df.
            hi_s = RATING_TO_SCORE[str(hi).strip().upper()]
            if hi_s is None:
                hi_s = df['rating_score'].min()
            lo_s = RATING_TO_SCORE[str(lo).strip().upper()]
            if lo_s is None:
                lo_s = df['rating_score'].max()
            # If scores increase as quality worsens, then band is [hi_s, lo_s] in numeric order.
            a, b = sorted([hi_s, lo_s])
            band_mask = (~df.get("is_cash", False)) & (df["rating_score"].astype(float).between(a, b, inclusive="both"))
            max_fail = _as_float(r.get("max_fail"))
            if max_fail is not None:
                constraints.append({
                    "type": "bucket_max",
                    "name": f"rating_band_{hi}_to_{lo}_max",
                    "mask": band_mask,
                    "alpha": max_fail,
                })
            target = _as_float(r.get("target"))
            if target is not None and target >= 0.999:
                # 100% in band => outside band must be 0
                outside_mask = (~df.get("is_cash", False)) & (~band_mask)
                constraints.append({
                    "type": "bucket_max",
                    "name": f"outside_rating_band_{hi}_to_{lo}_max0",
                    "mask": outside_mask,
                    "alpha": 0.0,
                })

        # average rating (soft by default with slack)
        avg_rules = [r for r in self._rules("ratings") if str(r.get("rule")) == "Average Rating"]
        if avg_rules and "rating_score" in df.columns:
            min_warn = avg_rules[0].get("min_warn")
            thr = RATING_TO_SCORE[str(min_warn).strip().upper()]
            if thr is not None:
                constraints.append({
                    "type": "avg_rating_max_soft", # here use 'max' as we're using rating score, the higher the worse rating
                    "name": "avg_rating_floor",
                    "threshold": float(thr),
                })

        # obligor constraints
        obligor_rules = self._rules("obligor")
        by_port_mv = [r for r in obligor_rules if str(r.get("rule")) == "Max Any Obligor by Port Market Value"]
        if by_port_mv and "obligor" in df.columns:
            chosen = _select_port_mv_band_rule(by_port_mv, mv_tot)
            cap = _as_float(chosen.get("max_fail")) if chosen else None
            if cap is not None:
                for obligor, sub in df.loc[~df.get("is_cash", False)].groupby("obligor"):
                    mask = (~df.get("is_cash", False)) & (df["obligor"] == obligor)
                    constraints.append({
                        "type": "bucket_max",
                        "name": f"obligor_{obligor}_max",
                        "mask": mask,
                        "alpha": cap,
                    })

        # max any obligor by rating (A+ or worse)
        by_ob_rating = [r for r in obligor_rules if str(r.get("rule")) == "Max Any Obligor by Rating"]
        if by_ob_rating:
            cap = _as_float(by_ob_rating[0].get("max_fail"))
            group = str(by_ob_rating[0].get("group") or "")
            label = _extract_rating_label(group)
            # interpret "A+ or worse" as rating_score >= score(A+)
            if cap is not None and label:
                thr = RATING_TO_SCORE[str(label).strip().upper()]
                if thr is not None:
                    for obligor in df.loc[~df.get("is_cash", False), "obligor"].unique():
                        mask = (~df.get("is_cash", False)) & (df["obligor"] == obligor) & (df["rating_score"].astype(float) >= thr)
                        if mask.any():
                            constraints.append({
                                "type": "bucket_max",
                                "name": f"obligor_{obligor}_{label}_or_worse_max",
                                "mask": mask,
                                "alpha": cap,
                            })

        # max any CUSIP5 by rating (A+ or worse)
        by_cusip5 = [r for r in obligor_rules if str(r.get("rule")) == "Max Any CUSIP5 by Rating"]
        if by_cusip5:
            cap = _as_float(by_cusip5[0].get("max_fail"))
            group = str(by_cusip5[0].get("group") or "")
            label = _extract_rating_label(group)
            if cap is not None and label:
                thr = RATING_TO_SCORE[str(label).strip().upper()]
                if thr is not None:
                    cusip5 = df["cusip"].astype(str).str.slice(0, 5)
                    for c5 in cusip5.unique():
                        mask = (~df.get("is_cash", False)) & (cusip5 == c5) & (df["rating_score"].astype(float) >= thr)
                        if mask.any():
                            constraints.append({
                                "type": "bucket_max",
                                "name": f"cusip5_{c5}_{label}_or_worse_max",
                                "mask": mask,
                                "alpha": cap,
                            })

        # sector cap (Any Other Sector) -> cap each single sector at max_fail
        sector_rules = self._rules("sector")
        sec_any_other = [r for r in sector_rules if str(r.get("rule")).strip() in {"Sector / Sub-Industry Structure", "Mapped Sector 2*"}]
        if sec_any_other and "sector" in df.columns:
            cap = _as_float(sec_any_other[0].get("max_fail"))
            if cap is not None:
                for sector in df.loc[~df.get("is_cash", False), "sector"].unique():
                    mask = (~df.get("is_cash", False)) & (df["sector"] == sector)
                    constraints.append({
                        "type": "bucket_max",
                        "name": f"sector_{sector}_max",
                        "mask": mask,
                        "alpha": cap,
                    })

        if "state" in df.columns:
            st = df["state"].astype(str).str.strip().str.upper()
            non_ca_mask = (~df.get("is_cash", False)) & (~st.isin({"CA", "US"}))

            constraints.append({
                "type": "bucket_max",
                "name": "state_non_ca_max_10pct",
                "mask": non_ca_mask,
                "alpha": 0.10,
            })

        return constraints, warnings

    def _constraint_report(self, df: pd.DataFrame, mv_tot: float, constraints: List[Dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for c in constraints:
            ctype = c.get("type")
            name = c.get("name", "")
            if ctype == "bucket_max":
                mask = c.get("mask")
                alpha = float(c.get("alpha"))
                if mask is None:
                    continue
                aligned_mask = (
                    mask.reindex(df.index, fill_value=False).astype(bool)
                    if isinstance(mask, pd.Series)
                    else pd.Series(False, index=df.index)
                )
                bucket_mv = float(df.loc[aligned_mask, "market_value"].sum())
                bucket_w = bucket_mv / mv_tot if mv_tot > 0 else 0.0
                rows.append({
                    "constraint": name,
                    "type": ctype,
                    "cap_alpha": alpha,
                    "actual_weight": bucket_w,
                    "excess": bucket_w - alpha,
                    "status": "VIOLATION" if bucket_w > alpha + 1e-10 else "OK",
                })
            elif ctype == "avg_rating_max_soft":
                thr = float(c.get("threshold"))
                if "rating_score" not in df.columns:
                    continue
                avg_score = float((df["rating_score"].astype(float) * df["market_value"].astype(
                    float)).sum()) / mv_tot if mv_tot > 0 else 0.0
                rows.append({
                    "constraint": name,
                    "type": ctype,
                    "cap_alpha": thr,
                    "actual_weight": avg_score,  # here it's an avg score, not a weight
                    "excess": avg_score - thr,
                    "status": "VIOLATION" if avg_score > thr + 1e-10 else "OK",
                })
        return pd.DataFrame(rows)


    # solve linear programming
    def run(self, df: pd.DataFrame) -> GlobalSolveResult:
        from base import BaseRuleSet, _to_df  # uses existing helper

        df0 = df.copy()
        key_cols = [c for c in ["cusip", "obligor"] if c in df0.columns]
        maturity_col = "maturity"
        maturity_map = None
        if maturity_col in df0.columns and len(key_cols) == 2:
            maturity_map = (
                df0[key_cols + [maturity_col]]
                .drop_duplicates(subset=key_cols, keep="first")
                .copy()
            )

        mv_tot = float(df0["market_value"].sum())
        df0["weight"] = df0["market_value"] / mv_tot if mv_tot > 0 else 0.0

        # Step 1: hard filters
        hard_sells = self._hard_sells(df0)
        # apply hard sells using BaseRuleSet helper
        df1 = BaseRuleSet._apply_trades_to_df(self, df0, hard_sells)

        # Step 2: build constraints and solve LP
        constraints, warn_info = self._build_constraints_spec(df1, mv_tot=mv_tot)
        pre_report = self._constraint_report(df1, mv_tot=mv_tot, constraints=constraints)

        if self.use_integer:
            lp_sells = self._solve_milp_whole_positions(df1, mv_tot=mv_tot, constraints=constraints)
        else:
            lp_sells = self._solve_lp(df1, mv_tot=mv_tot, constraints=constraints)
        df2 = BaseRuleSet._apply_trades_to_df(self, df1, lp_sells)

        post_report = self._constraint_report(df2, mv_tot=mv_tot, constraints=constraints)
        if maturity_map is not None and maturity_col in df2.columns:
            df2 = df2.drop(columns=[maturity_col]).merge(maturity_map, on=key_cols, how="left")

        executed = _to_df(hard_sells + lp_sells)
        warn_trades = _to_df([
            {
                "cusip": None,
                "obligor": None,
                "sell_mv": 0.0,
                "reason": f"Info: {w}",
                "old_rating": None,
                "weight": None,
                "rule": "Soft/Info",
            }
            for w in warn_info
        ])
        target_trades = _to_df([])
        return GlobalSolveResult(df2, executed, warn_trades, target_trades, pre_report, post_report)

    def _solve_lp(self, df: pd.DataFrame, mv_tot: float, constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Solve LP: minimize total sells subject to bucket constraints.

        We use SciPy's linprog so this works in lightweight environments.
        """
        from scipy.optimize import linprog

        # decision set: non-cash rows with positive MV
        decision_mask = ~df.get("is_cash", False)
        df_dec = df.loc[decision_mask & (df["market_value"] > 0)].copy()

        idx = list(df_dec.index)
        MV = df_dec["market_value"].astype(float).to_dict()

        # build linprog matrices
        n = len(idx)
        use_slack = any(c.get("type") == "avg_rating_max_soft" for c in constraints)
        bigM = 1e6

        # objective
        cvec = [1.0] * n + ([bigM] if use_slack else [])

        A_ub = []
        b_ub = []

        # we don't have this for now, but could setup in the future to add a upper limit for 'total sell cap'
        for c in constraints:
            if c.get("type") != "total_sell_cap":
                continue
            cap = float(c.get("cap", 0.0))
            cap = max(cap, 0.0)
            row = [1.0] * n + ([0.0] if use_slack else [])
            A_ub.append(row)
            b_ub.append(cap)

        # bucket max
        for c in constraints:
            if c.get("type") != "bucket_max":
                continue
            alpha = float(c.get("alpha"))
            mask = c.get("mask")
            if mask is None:
                continue
            G = [k for k, i in enumerate(idx) if bool(mask.loc[i])]
            if not G:
                continue
            const_term = sum(MV[idx[k]] for k in G) # total market_value in this maturity bucket
            # -sum_{i in G} s_i <= alpha * MVtot - const_term
            row = [0.0] * n
            for k in G:
                row[k] = -1.0
            if use_slack:
                row.append(0.0)
            A_ub.append(row)
            b_ub.append(alpha * mv_tot - const_term)

        # soft avg rating cap: -sum(score_i*s_i) - MVtot*slack <= thr*MVtot - const_term
        if use_slack:
            c0 = next(c for c in constraints if c.get("type") == "avg_rating_max_soft")
            thr = float(c0.get("threshold"))
            if "rating_score" in df_dec.columns:
                score = df_dec["rating_score"].astype(float).to_list()
                const_term = float((df_dec["rating_score"].astype(float) * df_dec["market_value"].astype(float)).sum())
                row = [-float(score[k]) for k in range(n)] + [-mv_tot]
                A_ub.append(row)
                b_ub.append(thr * mv_tot - const_term)

        bounds = [(0.0, float(MV[i])) for i in idx] + ([(0.0, None)] if use_slack else [])

        res = linprog(c=cvec, A_ub=A_ub or None, b_ub=b_ub or None, bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"Global LP infeasible or solver failed: {res.message}")

        x = res.x

        sells: List[Dict[str, Any]] = []
        for pos, i in enumerate(idx):
            val = float(x[pos])
            if val <= 0:
                continue
            if val < self.min_trade_mv:
                continue
            row = df_dec.loc[i]
            sells.append({
                "cusip": row.get("cusip"),
                "obligor": row.get("obligor"),
                "sell_mv": min(val, float(row.get("market_value"))),
                "reason": "Global LP: satisfy constraints with minimal total sells",
                "old_rating": row.get("rating"),
                "weight": row.get("weight"),
                "rule": "Global LP",
            })
        return sells

    def _solve_milp_whole_positions(self, df: pd.DataFrame, mv_tot: float, constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        MILP version: if we sell a bond, we sell the entire position.
        Decision y_i ∈ {0,1}; sell_mv_i = MV_i * y_i
        """
        import pulp

        decision_mask = ~df.get("is_cash", False)
        df_dec = df.loc[decision_mask & (df["market_value"] > 0)].copy()

        idx = list(df_dec.index)
        MV = df_dec["market_value"].astype(float).to_dict()

        # Optional: if a position is below min_trade_mv, disallow selling it
        sellable = {i: (MV[i] >= float(self.min_trade_mv)) for i in idx}

        use_slack = any(c.get("type") == "avg_rating_max_soft" for c in constraints)
        bigM = 1e6

        prob = pulp.LpProblem("GlobalRebalanceMILP", pulp.LpMinimize)

        y = {i: pulp.LpVariable(f"y_{i}", lowBound=0, upBound=1, cat=pulp.LpBinary) for i in idx}

        # If not sellable, force y_i = 0
        for i in idx:
            if not sellable[i]:
                prob += (y[i] == 0)

        slack = None
        if use_slack:
            slack = pulp.LpVariable("slack", lowBound=0, cat=pulp.LpContinuous)

        # Objective: minimize total MV sold, plus big penalty for slack
        prob += pulp.lpSum(MV[i] * y[i] for i in idx) + ((bigM * slack) if use_slack else 0)

        # total sell cap
        for c in constraints:
            if c.get("type") != "total_sell_cap":
                continue
            cap = max(float(c.get("cap", 0.0)), 0.0)
            prob += (pulp.lpSum(MV[i] * y[i] for i in idx) <= cap)

        # bucket max: -sum_{i in G} sell_mv_i <= alpha*MVtot - const_term
        for c in constraints:
            if c.get("type") != "bucket_max":
                continue
            alpha = float(c.get("alpha"))
            mask = c.get("mask")
            if mask is None:
                continue

            G = [i for i in idx if bool(mask.loc[i])]
            if not G:
                continue

            const_term = sum(MV[i] for i in G)
            prob += (-pulp.lpSum(MV[i] * y[i] for i in G) <= alpha * mv_tot - const_term)

        # soft avg rating cap (same algebra, but sell_mv_i = MV_i*y_i)
        if use_slack:
            c0 = next(c for c in constraints if c.get("type") == "avg_rating_max_soft")
            thr = float(c0.get("threshold"))

            if "rating_score" in df_dec.columns:
                score = df_dec["rating_score"].astype(float).to_dict()
                const_term = float((df_dec["rating_score"].astype(float) * df_dec["market_value"].astype(float)).sum())

                prob += (
                        -pulp.lpSum(score[i] * MV[i] * y[i] for i in idx) - mv_tot * slack
                        <= thr * mv_tot - const_term
                )

        solver = pulp.PULP_CBC_CMD(msg=False)
        status = prob.solve(solver)

        if pulp.LpStatus[status] not in ("Optimal", "Optimal Infeasible"):  # CBC sometimes reports edge cases oddly
            raise RuntimeError(f"Global MILP infeasible or solver failed: {pulp.LpStatus[status]}")

        sells: List[Dict[str, Any]] = []
        for i in idx:
            if pulp.value(y[i]) >= 0.5:
                row = df_dec.loc[i]
                sells.append({
                    "cusip": row.get("cusip"),
                    "obligor": row.get("obligor"),
                    "sell_mv": float(row.get("market_value")),
                    "reason": "Global MILP: whole-position sells to satisfy constraints with minimal total sells",
                    "old_rating": row.get("rating"),
                    "weight": row.get("weight"),
                    "rule": "Global MILP",
                })

        return sells

