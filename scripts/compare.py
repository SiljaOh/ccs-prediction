"""Collect the metrics of all models into one table and figure (report section 3.6)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ccs import plots  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
names = {"ridge": "Ridge", "rf": "Random Forest", "cnn": "CNN", "lstm": "LSTM"}
res = {}
for key, name in names.items():
    f = out / f"{key}_metrics.json"
    if f.exists():
        res[name] = json.loads(f.read_text())["test"]
print("| Model | R² | MAE (Å²) | Pearson r |\n|---|---|---|---|")
for n, m in res.items():
    print(f"| {n} | {m['r2']:.3f} | {m['mae']:.2f} | {m['pearson_r']:.3f} |")
plots.model_comparison(res, out / "figures" / "model_comparison.png")
