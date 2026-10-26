# Data

The ion mobility dataset (844,122 peptide measurements with sequence, modifications, charge, m/z,
CCS and Andromeda score) was provided by the Computational Mass Spectrometry group at TUM for the
course project and is **not included** in this repository.

The scripts expect a CSV with at least these columns:

| column | example |
|---|---|
| `Sequence` | `AAAAAAAAAAGAAGGR` |
| `Modified sequence` | `_(ac)AAAAAAAAAAGAAGGR_` |
| `Charge` | `2` |
| `CCS` | `413.32` |

Optional: `m/z`, `Score` (used to pick the best entry among near duplicates).
