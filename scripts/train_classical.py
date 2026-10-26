"""Train and evaluate Ridge Regression and Random Forest (report sections 2.3.1, 2.3.2)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from ccs import data, features, metrics, plots, splits  # noqa: E402
from ccs.models import classical  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("csv")
p.add_argument("--model", choices=["ridge", "rf"], required=True)
p.add_argument("--tune", action="store_true", help="run the full grid search")
p.add_argument("--fast", action="store_true", help="smaller Random Forest for quick runs")
p.add_argument("--cv", action="store_true", help="5-fold cross-validation on the training set")
p.add_argument("--out", default="results")
a = p.parse_args()
out = Path(a.out)
(out / "figures").mkdir(parents=True, exist_ok=True)

df = data.prepare(a.csv)
X, names = features.tabular(df)
y = df["CCS"].to_numpy()
tr, va, te = splits.split_indices(len(df))

model = classical.ridge(tune=a.tune) if a.model == "ridge" else classical.random_forest(tune=a.tune, fast=a.fast)
model.fit(X[tr], y[tr])
est = getattr(model, "best_estimator_", model)
if hasattr(model, "best_params_"):
    print("best parameters:", model.best_params_)

res = {"train": metrics.evaluate(y[tr], est.predict(X[tr])), "test": metrics.evaluate(y[te], est.predict(X[te]))}
if a.cv:
    res["cv_r2_mean"], res["cv_r2_std"] = classical.cross_validate(est, X[tr], y[tr])
print(json.dumps(res, indent=2))

title = {"ridge": "Ridge Regression", "rf": "Random Forest"}[a.model]
pred = est.predict(X[te])
plots.predicted_vs_experimental(y[te], pred, df["Charge"].to_numpy()[te], res["test"], title,
                                out / "figures" / f"{a.model}_pred_vs_exp.png")
plots.error_distribution(y[te], pred, f"{title}: error distribution (test set)",
                         out / "figures" / f"{a.model}_errors.png")
if a.model == "rf":
    imp = est.feature_importances_
else:
    imp = est.named_steps["ridge"].coef_  # standardised coefficients
groups = classical.grouped_importance(np.asarray(imp), names, features.feature_group)
res["feature_importance"] = groups
plots.feature_importance(groups, f"{title}: grouped feature importance", out / "figures" / f"{a.model}_importance.png")
(out / f"{a.model}_metrics.json").write_text(json.dumps(res, indent=2))
