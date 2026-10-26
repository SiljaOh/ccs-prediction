# Exploration (original project scripts)

My own experiments from the project (November 2025 – January 2026), kept as they were except for the data path:

- `lightgbm_baseline.py`: gradient boosting (LightGBM) on hand-crafted physicochemical sequence features
  (amino acid frequencies, hydropathy, aromaticity, charge densities, rough pI) plus charge state,
  evaluated on a 15 % hold-out set with charge-coloured plots.
- `lightgbm_report_plots.py`: tuned variant used to develop the plot style of the final talk.
  Note: it evaluates on the training split, so its numbers are not test-set results.

The cleaned-up pipeline for the four models compared in the report lives in `src/ccs` and `scripts/`.
