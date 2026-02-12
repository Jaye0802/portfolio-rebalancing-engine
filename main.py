"""Entry point: hard filters + global optimization.

Outputs
-------
portfolio_results.xlsx with sheets:
  - Updated Portfolio
  - Executed Trades
  - Warning Trades
  - Target Trades
  - Constraint Check (Pre)
  - Constraint Check (Post)
"""

from __future__ import annotations

import pandas as pd

from data_loader import load_portfolio
from rules_config import read_rules_table
from global_optimizer import GlobalRebalanceEngine


PORTFOLIO_XLSX = "Bonds.xlsx"
RULES_XLSX = "rules_table.xlsx"
USE_INTEGER_TRADES = True # if use integer programming or not


def main() -> None:
    df_raw, cash = load_portfolio(PORTFOLIO_XLSX)
    rules_config, _standard_sections = read_rules_table(RULES_XLSX)
    strategy_name = list(rules_config.keys())[0]

    # Minimum trade market value threshold (optional: set to 0 for pure math solution)
    min_trade_mv = 5_000.0

    engine = GlobalRebalanceEngine(
        strategy_name=strategy_name,
        rule_config=rules_config,
        min_trade_mv=min_trade_mv,
        forbidden_sectors=[],  # put the 'remove immediately' sectors here
        use_integer = USE_INTEGER_TRADES
    )

    result = engine.run(df_raw)


    # Summary sheets: Initial vs After weights

    mv_tot = float(df_raw["market_value"].sum())  # fixed denominator (incl cash)

    def rule_group_sheet(col: str):

        def one(df):
            d = df.loc[~df.get("is_cash", False)].copy()
            s = d.groupby(col, dropna=False)["market_value"].sum() / mv_tot
            return s.rename_axis(col).to_frame(name="weights")

        out = one(df_raw).reset_index().merge(
            one(result.updated_portfolio).reset_index(),
            on=col,
            how="outer",
            suffixes=("_Initial", "_After"),
        ).fillna(0.0).rename(columns={"weight_Initial": "Initial", "weight_After": "After"})

        return out

    # Write sheets (rule groups as index/rows, plus Any Other)
    if USE_INTEGER_TRADES:
        output_path = f"portfolio_results_integer_integer.xlsx"
    else:
        output_path = "portfolio_results_lp.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        result.updated_portfolio.to_excel(writer, sheet_name="Updated Portfolio", index=False)
        result.executed_trades.to_excel(writer, sheet_name="Executed Trades", index=False)
        result.warning_trades.to_excel(writer, sheet_name="Warning Trades", index=False)
        result.target_trades.to_excel(writer, sheet_name="Target Trades", index=False)
        result.constraint_report_pre.to_excel(writer, sheet_name="Constraint Check (Pre)", index=False)
        result.constraint_report_post.to_excel(writer, sheet_name="Constraint Check (Post)", index=False)
        rule_group_sheet("maturity_year").to_excel(writer, sheet_name="Maturity", index=False)
        rule_group_sheet("rating").to_excel(writer, sheet_name="Rating", index=False)
        rule_group_sheet("holdings_sector").to_excel(writer, sheet_name="Sector", index=False)
        rule_group_sheet("obligor").to_excel(writer, sheet_name="Obligor", index=False)
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
