"""parse Std Management Style Rules into Python dicts."""

from __future__ import annotations

import pandas as pd
from typing import Dict, Any, Tuple, Set

from base import _standardize_columns, _standardize_name


def read_rules_table(path) -> Tuple[Dict[str, Any], Set[str]]:
    """Parse the Std Management Style Rules Excel file into a nested dictionary."""
    df = pd.read_excel(path)
    df_copy = df.copy().dropna(axis=1, how='all')
    df_copy = _standardize_columns(df_copy)
    col0 = df_copy.columns[0]

    # extract strategy name like "Franklin Muni Ladder 1-7 Year NA"
    first_row = df_copy.iloc[0]
    strategy_name = _standardize_name(first_row.dropna().iloc[0])

    # detect section header rows (Cash, Coupon, Duration, ...)
    is_section_header = df_copy[col0].notna() & df_copy.drop(columns=[col0]).isna().all(axis=1)
    section_names = df_copy.loc[is_section_header, col0].tolist()
    standard_rules_set = set([s.strip().lower().replace(" ", "_") for s in section_names])

    # map section name -> (start_row_idx, end_row_idx) [end exclusive]
    section_indices = df_copy.index[is_section_header].tolist()

    # identify rows belong to this section
    section_ranges = {}
    for idx, start_header_row in enumerate(section_indices):
        name = _standardize_name(str(df_copy.loc[start_header_row, col0]))
        start = start_header_row + 1
        if idx + 1 < len(section_indices):
            end = section_indices[idx + 1]
        else:
            end = len(df_copy)
        section_ranges[name] = (start, end)

    rules_config: Dict[str, Any] = {strategy_name: {}}

    # for each section, decide which columns to keep: if non-all-NaN inside that column, then store rows using only those columns.
    for section_name, (start, end) in section_ranges.items():
        if section_name == strategy_name:
            continue

        section_df = df_copy.iloc[start:end]
        if section_df.empty:
            rules_config[strategy_name][section_name] = []
            continue

        non_empty_cols = section_df.drop(columns=[col0]).columns[section_df.drop(columns=[col0]).notna().any(axis=0)].tolist()
        cols_to_keep = [col0] + non_empty_cols

        section_rules = []
        for _, row in section_df.iterrows():

            if row[cols_to_keep].drop(labels=[col0]).isna().all():
                continue
            section_rules.append(row[cols_to_keep].to_dict())

        rules_config[strategy_name][section_name] = section_rules
    return rules_config, standard_rules_set


if __name__ == "__main__":
    rules_dict, rules_name = read_rules_table("Std Management Style Rules (exported on 11_26_2025).xlsx")
    print(rules_name)