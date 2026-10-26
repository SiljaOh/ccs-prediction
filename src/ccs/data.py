"""Loading and cleaning of the ion mobility peptide dataset.

Expected input: a CSV with at least the columns
``Sequence``, ``Modified sequence``, ``Charge`` and ``CCS``
(MaxQuant-style modified sequences such as ``_(ac)AAM(ox)PEK_``).
Optional columns ``m/z`` and ``Score`` (Andromeda score) are used if present.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

MOD_PATTERN = re.compile(r"\(([^)]+)\)")
MAX_LENGTH = 32  # upper IQR limit of the peptide length distribution (see report, Fig. 2)


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Sequence", "Modified sequence", "Charge", "CCS"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identical rows, then keep one representative per near-duplicate group.

    Near duplicates share sequence, modification and charge but differ in CCS.
    If an Andromeda ``Score`` is available, the highest scoring entry is kept,
    otherwise the median CCS of the group is used.
    """
    df = df.drop_duplicates()
    keys = ["Modified sequence", "Charge"]
    if "Score" in df.columns:
        df = df.sort_values("Score", ascending=False).drop_duplicates(keys)
    else:
        df = df.groupby(keys, as_index=False).agg(
            {"Sequence": "first", "CCS": "median", **({"m/z": "first"} if "m/z" in df else {})}
        )
    return df.reset_index(drop=True)


def iqr_upper_limit(lengths: pd.Series) -> float:
    q1, q3 = np.percentile(lengths, [25, 75])
    return q3 + 1.5 * (q3 - q1)


def filter_length(df: pd.DataFrame, max_length: int = MAX_LENGTH) -> pd.DataFrame:
    return df[df["Sequence"].str.len() <= max_length].reset_index(drop=True)


def modifications(modified_sequence: str) -> list[str]:
    return MOD_PATTERN.findall(modified_sequence)


def prepare(path: str) -> pd.DataFrame:
    """Full cleaning pipeline used for all models."""
    df = load(path)
    df = remove_duplicates(df)
    df = filter_length(df)
    df["Charge"] = df["Charge"].astype(int)
    return df
