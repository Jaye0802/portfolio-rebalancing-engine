"""Base module for data processing utilities."""

from typing import Any, Dict, List
from abc import ABC, abstractmethod

import re
import pandas as pd
from typing import Optional

from consts import _NOT_RATED, _MOODYS_TO_SP, RETURN_COLS


def _choose_best_rating(row, rating_scale):
    ratings = [
        row.get("moody_to_sp_rating"),
        row.get("s&p_rating"),
        row.get("fitch_rating"),
    ]

    if "U.S." in ratings:
        return "U.S.", None

    valid = [r for r in ratings if r in rating_scale]
    if not valid:
        return "NR", None

    best = min(valid, key=lambda r: rating_scale[r])
    return best, rating_scale[best]

def _parse_moodys_rating(x) -> Optional[str]:
    """
    Parse a Moody's long-term rating (e.g., 'A3', 'Baa2', 'Aa1', 'Aaa', 'NR')
    and return an S&P-style long-term rating (e.g., 'A-', 'BBB', 'AA+', 'AAA').
    """
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
        return None

    s = str(x).strip()
    if not s:
        return None

    s_up = s.upper().strip()
    s_up = re.sub(r"\s+", "", s_up)

    if s_up in {"US", "U.S."}:
        return "U.S."

    if s_up in _NOT_RATED:
        return None

    # Common variants: sometimes there'll be "A-3" or "BAA-2" -> normalize by removing non-alnum
    s_norm = re.sub(r"[^A-Z0-9]", "", s_up)

    out = _MOODYS_TO_SP.get(s_norm)
    return out

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy.columns = (
        df_copy.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    return df_copy

def _standardize_name(name) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _to_df(items: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(items) if items else pd.DataFrame(columns=RETURN_COLS)


class BaseRuleSet(ABC):
    """Abstract base class for a rule-set handler"""
    def __init__(self, strategy_name: str, section_header: str, config: Dict[str, Any]):
        self.strategy_name = strategy_name
        self.section_header = section_header
        self.config = config
        self.messages: List[str] = []

    @property
    def rules(self) -> List[Dict[str, Any]]:
        get_strategy = self.config.get(self.strategy_name, {})
        if not get_strategy:
            raise ValueError(f"No strategy found for {self.strategy_name} ")
        get_rule = get_strategy.get(self.section_header, [])
        if not get_rule:
            raise ValueError(f"No rules found for {self.strategy_name}{self.section_header} ")
        return get_rule

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a trades DataFrame with at least:
        ['CUSIP','SellMV','Reason','Rule', ...]
        """
        ...

    def _apply_trades_to_df(self, df: pd.DataFrame, sells: List[Dict[str, Any]]) -> pd.DataFrame:
        if not sells:
            return df.copy()

        df_new = df.copy()

        trades_df = pd.DataFrame(sells)
        identifier_cols = ["cusip", "obligor"]
        sell_by_key = (trades_df.groupby(identifier_cols)["sell_mv"].sum()
                       .reset_index()
                       .rename(columns={"sell_mv": "total_sell_mv"}))

        # original total MV (constant baseline)
        total_mv0 = float(df_new["market_value"].sum())

        df_new = df_new.merge(sell_by_key, on=identifier_cols, how="left")
        df_new["total_sell_mv"] = df_new["total_sell_mv"].fillna(0.0)

        # total sold (to be added to cash)
        sold_total = float(df_new["total_sell_mv"].sum())
        df_new["market_value"] = df_new["market_value"] - df_new["total_sell_mv"]

        # drop fully sold positions
        df_new = df_new[df_new["market_value"] > 0].copy()

        # add sold_total to cash
        cash_mask = df_new.get("is_cash", False)
        if isinstance(cash_mask, bool):
            cash_mask = pd.Series([False] * len(df_new), index=df_new.index)

        if cash_mask.any():
            # if multiple cash rows, add to the first one
            cash_idx = df_new.index[cash_mask][0]
            df_new.loc[cash_idx, "market_value"] = df_new.loc[cash_idx, "market_value"] + sold_total
        else:
            cash_row = {c: None for c in df_new.columns}
            cash_row.update({"cusip": "CASH", "obligor": "CASH", "market_value": sold_total, "is_cash": True})
            df_new = pd.concat([df_new, pd.DataFrame([cash_row])], ignore_index=True)

        # recompute weights using constant total MV
        if total_mv0 > 0:
            df_new["weight"] = df_new["market_value"] / total_mv0
        else:
            df_new["weight"] = 0.0

        df_new.drop(columns=["total_sell_mv"], inplace=True, errors="ignore")
        return df_new