import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ccs import data, features, metrics, splits  # noqa: E402


def toy():
    return pd.DataFrame({
        "Sequence": ["PEPTIDEK", "AMLKR", "PEPTIDEK"],
        "Modified sequence": ["_PEPTIDEK_", "_(ac)AM(ox)LKR_", "_PEPTIDEK_"],
        "Charge": [2, 3, 2],
        "CCS": [350.0, 400.0, 352.0],
    })


def test_one_hot_shape_and_padding():
    x = features.one_hot("ACD")
    assert x.shape == (32, 20)
    assert x.sum() == 3 and x[3:].sum() == 0


def test_modifications_parsed():
    assert data.modifications("_(ac)AM(ox)LKR_") == ["ac", "ox"]
    assert features.modification_flags(["ox"]).tolist() == [1.0, 0.0, 1.0]


def test_mass_and_mz():
    # PEPTIDE monoisotopic mass is 799.36 Da
    m = features.peptide_mass("PEPTIDE", [])
    assert abs(m - 799.3600) < 1e-3
    assert abs(features.mz(m, 2) - (m + 2 * features.PROTON) / 2) < 1e-9


def test_tabular_feature_count():
    X, names = features.tabular(toy())
    assert X.shape == (3, 5 + 20 + 640) and len(names) == X.shape[1]


def test_near_duplicates_collapsed():
    assert len(data.remove_duplicates(toy())) == 2


def test_split_proportions():
    tr, va, te = splits.split_indices(1000)
    assert (len(tr), len(va), len(te)) == (750, 100, 150)
    assert len(set(tr) | set(va) | set(te)) == 1000


def test_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    m = metrics.evaluate(y, y)
    assert m["mae"] == 0 and m["r2"] == 1
