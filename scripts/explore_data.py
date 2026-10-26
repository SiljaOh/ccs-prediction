"""Dataset statistics and distribution plots (report section 3.1)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ccs import data, plots  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("csv")
p.add_argument("--out", default="results")
a = p.parse_args()
out = Path(a.out)
(out / "figures").mkdir(parents=True, exist_ok=True)

raw = data.load(a.csv)
dedup = data.remove_duplicates(raw)
lengths = dedup["Sequence"].str.len()
limit = data.iqr_upper_limit(lengths)
final = data.filter_length(dedup)

stats = {
    "rows_raw": len(raw),
    "rows_after_deduplication": len(dedup),
    "iqr_length_limit": float(limit),
    "share_longer_than_limit": float((lengths > data.MAX_LENGTH).mean()),
    "rows_final": len(final),
    "charge_states": {int(k): int(v) for k, v in final["Charge"].value_counts().sort_index().items()},
    "ccs_median": float(final["CCS"].median()),
    "ccs_mean": float(final["CCS"].mean()),
    "share_modified": float(final["Modified sequence"].str.contains(r"\(").mean()),
}
print(json.dumps(stats, indent=2))
(out / "data_stats.json").write_text(json.dumps(stats, indent=2))
plots.ccs_distribution(final["CCS"].to_numpy(), out / "figures" / "ccs_distribution.png")
plots.length_distribution(lengths.to_numpy(), limit, out / "figures" / "length_distribution.png")
