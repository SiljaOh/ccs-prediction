# Machine Learning vs. Deep Learning for Peptide CCS Prediction

Can classical machine learning keep up with deep learning when predicting the **collision cross section (CCS)** of peptides from their sequence? This project compares four models on an ion mobility dataset of **844,122 peptide measurements** (240,000+ unique peptides after cleaning):

**Ridge Regression** · **Random Forest** · **1D-CNN** · **LSTM**

The project was carried out from February 2025 to January 2026 in the **Computational Mass Spectrometry group of Prof. Mathias Wilhelm (TUM)**, together with Lara Groß and Anna Niederlechner.

## Why CCS?

Ion mobility spectrometry separates ions by their size and shape in a drift gas. The resulting CCS value is a fourth identifier for a peptide, in addition to retention time, m/z and fragment spectrum. It helps to tell apart isobaric and isomeric peptides that mass spectrometry alone cannot separate. Measuring CCS requires special instruments, so accurate predictions from sequence can add this dimension to any proteomics dataset.

## Results (test set, from our report)

| Model | R² | MAE (Å²) | Pearson r |
|---|---|---|---|
| Ridge Regression | 0.963 | 14.41 | |
| Random Forest | 0.971 | 12.44 | |
| CNN | 0.976 | 12.25 | 0.988 |
| **LSTM** | **0.977** | **11.80** | **0.988** |

- **Both deep learning models are best**, and they are practically tied. This suggests that CCS is driven mostly by local sequence motifs and charge rather than long-range dependencies, so the cheaper CNN is a good choice at proteome scale.
- **The Random Forest comes close** (R² 0.971) at a fraction of the cost. Total mass (≈ 43 %) and charge (≈ 30 %) dominate its feature importances.
- **Even the linear Ridge baseline reaches R² 0.963**, which shows how strongly CCS scales with mass and charge. The errors that remain are mostly in charge states +3 and +4, where peptides are more flexible.
- **Modifications** (methionine oxidation, N-terminal acetylation) added almost nothing once the sequence and charge were known.

## About this code

The code in `src/` and `scripts/` is a clean re-implementation of the methods from our report, written in October 2026 so that the full pipeline is in one place and reproducible. My original experiments from the project are in [`exploration/`](exploration/). Results of re-running the pipeline will be added to `results/`.

## Method

```
peptide sequence ─┬─► one-hot (32 × 20) ──────────────────────────┬─► Ridge / Random Forest
modifications  ───┤   + amino acid counts, hydrophobic count,      │
charge, m/z    ───┘     length, total mass, m/z, charge  (≤ 690)   │
                                                                   │
one-hot sequence (32 × 20) ─► LSTM (64)  or  Conv(64,k5) + 2 residual blocks + avg-pool ─┐
charge (one-hot) ─► dense ────────────────────────────────────────────────────────────────┼─► dense ─► CCS
modification flags ─► dense ──────────────────────────────────────────────────────────────┘
```

**Data cleaning** (`src/ccs/data.py`): exact duplicates are removed. Near duplicates (same peptide, modification and charge, but different CCS) are collapsed to the entry with the best Andromeda score. Peptides longer than 32 residues are dropped; 32 is the IQR upper limit of the length distribution and removes 1.7 % of the data.

**Split** (`src/ccs/splits.py`): 75 % training, 10 % validation, 15 % test, with 5-fold cross-validation on the training set.

**Models** (`src/ccs/models/`):
- Ridge: standardised features, λ chosen by grid search (best 0.01)
- Random Forest: grid search over 600–1000 trees, depth 30–70, min. samples per leaf 2–10 and `max_features` ∈ {sqrt, 0.5}. Best: 1000 trees, depth 55, 6 samples per leaf, 0.5.
- LSTM: 64 hidden units, dropout 0.2, Adam (lr 1e-3), MSE on min-max scaled CCS, 50 epochs
- CNN: residual 1D-CNN, AdamW (lr 1e-3, weight decay 1e-4), linear warmup, ReduceLROnPlateau, early stopping (max. 45 epochs)

## Usage

```bash
pip install -r requirements.txt

python scripts/explore_data.py data/ccs.csv                     # dataset statistics + plots
python scripts/train_classical.py data/ccs.csv --model ridge --tune --cv
python scripts/train_classical.py data/ccs.csv --model rf       # add --tune for the full grid search
python scripts/train_deep.py data/ccs.csv --model lstm
python scripts/train_deep.py data/ccs.csv --model cnn
python scripts/compare.py                                       # table + comparison figure
pytest -q
```

The dataset is not public; see [`data/README.md`](data/README.md) for the expected format.

## Repository structure

```
src/ccs/           data cleaning, features, splits, metrics, plots, models (Ridge, RF, LSTM, CNN)
scripts/           command-line entry points for each step
tests/             unit tests (features, splits, metrics, model shapes)
exploration/       my original LightGBM experiments from the project
results/           metrics (JSON) and figures produced by the scripts
```

## Team

Lara Groß, Anna Niederlechner, Silja Ohlsen. Supervised in the Computational Mass Spectrometry group (Prof. Dr. Mathias Wilhelm), Technical University of Munich.
