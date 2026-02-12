"""Read bond portfolio data and compute metrics needed for rebalancing framework"""

from __future__ import annotations

from typing import Dict, Tuple, Optional
import pandas as pd

from base import _choose_best_rating, _parse_moodys_rating, _standardize_columns
from consts import RATING_TO_SCORE


def load_portfolio(
    bonds_path: str,
    codes_sheet: Optional[str] = "Codes",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Load the bond portfolio from Excel and compute:
    - market_value
    - weight
    - maturity_year
    - rating_score

    Also derives a simple cash dict with t_date_cash from rows whose obligor == 'CASH'.
    """

    #   Load portfolio sheet
    raw_df = pd.read_excel(bonds_path)
    df = _standardize_columns(raw_df)
    df['port_mgmt_style'] = "Style"
    num_cols = df.select_dtypes(include="number").columns
    df = df[(df[num_cols] >= 0).all(axis=1)]

    #   moody's to S&P rating
    if "moody's_rating" in df.columns:
        df["moody_to_sp_rating"] = df["moody's_rating"].apply(_parse_moodys_rating)

    #   Identify cash rows  
    df["is_cash"] = False
    for col in ["obligor", "holdings_sector"]:
        if col in df.columns:
            df["is_cash"] |= df[col].astype(str).str.upper().eq("CASH")
    if "cusip" in df.columns:
        df["is_cash"] |= df["cusip"].astype(str).str.upper().str.contains("CASH")

    #   Load rating scale if Codes sheet exists  
    rating_scale = RATING_TO_SCORE

    #   Market value  
    df["market_value"] = df["market_value_+_accrued"]

    total_mv = df["market_value"].sum()
    df["weight"] = df["market_value"] / total_mv

    #   Maturity year  
    if "maturity" not in df.columns:
        raise ValueError("Maturity column not found.")

    df["maturity"] = pd.to_datetime(df["maturity"])
    df.loc[~df["is_cash"], "maturity_year"] = (
        df.loc[~df["is_cash"], "maturity"].dt.year.astype(int)
    )
    df["maturity_year"] = df["maturity_year"].astype("Int64")

    #   Rating score  
    if rating_scale is not None:
        out = df.apply(lambda r: _choose_best_rating(r, rating_scale), axis=1, result_type="expand")
        df["rating"] = out[0]
        df["rating_score"] = out[1]

    #   Cash calculation  
    if "obligor" in df.columns:
        cash_mask = df["obligor"].astype(str).str.upper() == "CASH"
    elif "holdings_sector" in df.columns:
        cash_mask = df["sector"].astype(str).str.upper() == "CASH"

    cash_market_value = df.loc[cash_mask, "market_value"].sum()

    cash = {
        "t_date_cash": float(cash_market_value),
        "s_date_cash": float(cash_market_value),
    }

    return df, cash
