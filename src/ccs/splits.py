"""75 / 10 / 15 train-validation-test split (25 % held out, then split 40:60)."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42


def split_indices(n: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n)
    train, temp = train_test_split(idx, test_size=0.25, random_state=seed)
    val, test = train_test_split(temp, test_size=0.60, random_state=seed)
    return train, val, test
