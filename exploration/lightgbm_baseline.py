import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor

# =========================================
# Load Data
# =========================================
import sys
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/ccs_with_mods.csv"

df = pd.read_csv(DATA_PATH)

seqs = df["Sequence"].astype(str).values
charges = df["Charge"].astype(float).values
targets = df["CCS"].astype(float).values

# =========================================
# Feature Engineering
# =========================================

alphabet = sorted(list({aa for s in seqs for aa in s}))

hydropathy = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9,
    "R": -4.5
}

mass = {
    "A": 89, "R": 174, "N": 132, "D": 133, "C": 121, "Q": 146, "E": 147,
    "G": 75, "H": 155, "I": 131, "L": 131, "K": 146, "M": 149, "F": 165,
    "P": 115, "S": 105, "T": 119, "W": 204, "Y": 181, "V": 117
}

# pKa for isoelectric point estimation
pka = {"Cterm": 3.1, "Nterm": 8.0, "C": 8.5, "D": 3.9, "E": 4.1,
       "H": 6.0, "Cys": 8.3, "Tyr": 10.1, "Lys": 10.5, "Arg": 12.5}

def estimate_pI(seq):
    # super rough pI estimate using Henderson–Hasselbalch
    positives = seq.count("K") + seq.count("R") + seq.count("H")
    negatives = seq.count("D") + seq.count("E")
    return 7.0 + 0.3 * (positives - negatives)

def compute_features(seq):
    L = len(seq)

    # Amino acid frequencies
    freq = np.array([seq.count(a) / L for a in alphabet])

    # Hydrophobicity
    hyd = np.mean([hydropathy.get(a, 0) for a in seq])

    # Molecular weight
    avg_mass = np.mean([mass.get(a, 110) for a in seq])

    # Aromatic content
    aromaticity = (seq.count("F") + seq.count("W") + seq.count("Y")) / L

    # Positive charge density
    pos = (seq.count("K") + seq.count("R") + seq.count("H")) / L

    # Negative charge density
    neg = (seq.count("D") + seq.count("E")) / L

    # Isoelectric point
    pI = estimate_pI(seq)

    return np.hstack([freq, L, hyd, avg_mass, aromaticity, pos, neg, pI])


X_seq = np.vstack([compute_features(s) for s in seqs])

# Add charge as explicit feature
X = np.column_stack([X_seq, charges])

# =========================================
# Train/Test Split
# =========================================
X_train, X_test, y_train, y_test, charge_train, charge_test = train_test_split(
    X, targets, charges, test_size=0.15, random_state=42
)

# =========================================
# LightGBM tuned model
# =========================================
model = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.018,
    num_leaves=150,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.5,
    min_data_in_leaf=15
)

model.fit(X_train, y_train)

# =========================================
# Predictions & Metrics
# =========================================
preds = model.predict(X_test)

R2 = r2_score(y_test, preds)
MAE = mean_absolute_error(y_test, preds)
STD = np.std(preds - y_test)

print(f"R² = {R2:.4f}")
print(f"MAE = {MAE:.3f}")
print(f"STD Error = {STD:.3f}")

# =========================================
# BEAUTIFUL PLOTS with charge coloring
# =========================================
sns.set(style="whitegrid", font_scale=1.3)

plt.figure(figsize=(15, 6))

# ----- Error Distribution -----
plt.subplot(1, 2, 1)
sns.kdeplot(preds - y_test, fill=True, alpha=0.7, linewidth=2)
plt.axvline(0, color="black", linestyle="--")
plt.title(f"Error Distribution\nSTD Error = {STD:.2f}")
plt.xlabel("Predicted - True")

# ----- Predicted vs True with Charge Coloring -----
plt.subplot(1, 2, 2)
scatter = plt.scatter(
    y_test,
    preds,
    c=charge_test,
    cmap="viridis",
    s=40,
    alpha=0.7,
    edgecolor="black",
)

cbar = plt.colorbar(scatter)
cbar.set_label("Charge State")

lims = [min(y_test), max(y_test)]
plt.plot(lims, lims, "r--", linewidth=2)

plt.xlabel("True CCS")
plt.ylabel("Predicted CCS")
plt.title(f"Predicted vs True CCS\nR² = {R2:.3f}, MAE = {MAE:.1f}")

plt.tight_layout()
plt.show()
