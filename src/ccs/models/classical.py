"""Machine learning models: Ridge Regression baseline and Random Forest."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

# search spaces from the report (section 2.3)
RIDGE_GRID = {"ridge__alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100]}
RF_GRID = {
    "n_estimators": [600, 800, 1000],
    "max_depth": [30, 45, 55, 70],
    "min_samples_leaf": [2, 6, 10],
    "max_features": ["sqrt", 0.5],
}
# best configuration found in the report
RF_BEST = {"n_estimators": 1000, "max_depth": 55, "min_samples_leaf": 6, "max_features": 0.5}


def ridge(tune: bool = True):
    model = make_pipeline(StandardScaler(), Ridge(alpha=0.01))
    if tune:
        model = GridSearchCV(model, RIDGE_GRID, cv=5, scoring="r2", n_jobs=1)
    return model


def random_forest(tune: bool = False, fast: bool = False):
    params = dict(RF_BEST)
    if fast:  # for quick runs on a laptop
        params.update(n_estimators=100, max_depth=30)
    model = RandomForestRegressor(bootstrap=True, random_state=SEED, n_jobs=-1, **params)
    if tune:
        model = GridSearchCV(RandomForestRegressor(bootstrap=True, random_state=SEED, n_jobs=-1),
                             RF_GRID, cv=3, scoring="r2", n_jobs=1, verbose=1)
    return model


def cross_validate(model, X: np.ndarray, y: np.ndarray, folds: int = 5) -> tuple[float, float]:
    scores = cross_val_score(model, X, y, cv=KFold(folds, shuffle=True, random_state=SEED), scoring="r2")
    return float(scores.mean()), float(scores.std())


def grouped_importance(values: np.ndarray, names: list[str], group) -> dict[str, float]:
    """Sum feature importances (or |standardised coefficients|) per feature group, normalised to 1."""
    out: dict[str, float] = {}
    for v, n in zip(np.abs(values), names):
        out[group(n)] = out.get(group(n), 0.0) + float(v)
    total = sum(out.values()) or 1.0
    return dict(sorted(((k, v / total) for k, v in out.items()), key=lambda kv: -kv[1]))
