"""Train and evaluate the LSTM and CNN models (report sections 2.3.3, 2.3.4)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ccs import data, features, metrics, plots, splits  # noqa: E402
from ccs.models.deep import CNNRegressor, LSTMRegressor  # noqa: E402
from ccs.train_deep import MinMax, fit, loader, predict  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("csv")
p.add_argument("--model", choices=["lstm", "cnn"], required=True)
p.add_argument("--epochs", type=int, default=None, help="default: 50 (LSTM) / 45 (CNN) as in the report")
p.add_argument("--batch-size", type=int, default=256)
p.add_argument("--out", default="results")
a = p.parse_args()
torch.manual_seed(42)
out = Path(a.out)
(out / "figures").mkdir(parents=True, exist_ok=True)

df = data.prepare(a.csv)
seq, charge, mods = features.sequence_tensors(df)
y = df["CCS"].to_numpy()
tr, va, te = splits.split_indices(len(df))
scaler = MinMax().fit(y[tr])
yn = scaler.transform(y)

model = LSTMRegressor() if a.model == "lstm" else CNNRegressor()
epochs = a.epochs or (50 if a.model == "lstm" else 45)
history = fit(model,
              loader(seq[tr], charge[tr], mods[tr], yn[tr], a.batch_size, shuffle=True),
              loader(seq[va], charge[va], mods[va], yn[va], 1024),
              kind=a.model, epochs=epochs)

pred = scaler.inverse(predict(model, loader(seq[te], charge[te], mods[te], batch_size=1024)))
res = {"epochs_trained": len(history["train_loss"]), "test": metrics.evaluate(y[te], pred), "history": history}
print(json.dumps(res["test"], indent=2))

title = {"lstm": "LSTM", "cnn": "CNN"}[a.model]
plots.predicted_vs_experimental(y[te], pred, df["Charge"].to_numpy()[te], res["test"], title,
                                out / "figures" / f"{a.model}_pred_vs_exp.png")
plots.error_distribution(y[te], pred, f"{title}: error distribution (test set)", out / "figures" / f"{a.model}_errors.png")
plots.loss_curves(history, f"{title}: training curve", out / "figures" / f"{a.model}_loss.png")
torch.save(model.state_dict(), out / f"{a.model}.pt")
(out / f"{a.model}_metrics.json").write_text(json.dumps(res, indent=2))
