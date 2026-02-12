"""Project constants.

The uploaded code references a shared `consts.py` (not included in the
upload). This file provides the minimal set of constants used by:

* `base.py`
* `ratings.py`
* `maturity.py`

If your internal project already has a richer `consts.py`, you can ignore
this file or merge the needed pieces.
"""

from __future__ import annotations

from typing import Dict, List, Literal

_MOODYS_TO_SP = {
    "AAA": "AAA",
    "Aaa".upper(): "AAA",
    "AA1": "AA+",
    "AA2": "AA",
    "AA3": "AA-",
    "A1": "A+",
    "A2": "A",
    "A3": "A-",
    "BAA1": "BBB+",
    "BAA2": "BBB",
    "BAA3": "BBB-",
    "BA1": "BB+",
    "BA2": "BB",
    "BA3": "BB-",
    "B1": "B+",
    "B2": "B",
    "B3": "B-",
    "CAA1": "CCC+",
    "CAA2": "CCC",
    "CAA3": "CCC-",
    "CA": "CC",
    "C": "C",
    "D": "D",
}

_NOT_RATED = {"", "NR", "NA", "N/A", "NOT RATED", "UNRATED"}

RATING_ORDER: List[str] = [
    "AAA+","AAA", "AA+", "AA", "AA-", "A+","A","A-",
    "BBB+","BBB","BBB-",
    "BB+","BB","BB-",
    "B+","B","B-",
    "CCC","CC","C","D", "NR"
]

RATING_TO_SCORE: Dict[str, int] = {r: i for i, r in enumerate(RATING_ORDER)}

RETURN_COLS = ["cusip", "obligor", "sell_mv", "reason", "old_rating", "weight", "rule"]
