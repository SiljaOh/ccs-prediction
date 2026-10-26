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
charges = df["Charge"].astype(int).values
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

def estimate_pI(seq):
    positives = seq.count("K") + seq.count("R") + seq.count("H")
    negatives = seq.count("D") + seq.count("E")
    return 7.0 + 0.32 * (positives - negatives)

def compute_features(seq):
    L = len(seq)
    freq = np.array([seq.count(a) / L for a in alphabet])
    hyd = np.mean([hydropathy.get(a, 0) for a in seq])
    avg_mass = np.mean([mass.get(a, 110) for a in seq])
    aromaticity = (seq.count("F") + seq.count("W") + seq.count("Y")) / L
    pos = (seq.count("K") + seq.count("R") + seq.count("H")) / L
    neg = (seq.count("D") + seq.count("E")) / L
    pI = estimate_pI(seq)
    return np.hstack([freq, L, hyd, avg_mass, aromaticity, pos, neg, pI])

X_seq = np.vstack([compute_features(s) for s in seqs])
X = np.column_stack([X_seq, charges])

# =========================================
# Training split
# =========================================
X_train, _, y_train, _, charge_train, _ = train_test_split(
    X, targets, charges, test_size=0.15, random_state=42
)

# =========================================
# Model
# =========================================
model = LGBMRegressor(
    n_estimators=750,
    learning_rate=0.025,
    num_leaves=220,
    max_depth=-1,
    min_data_in_leaf=8,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42
)

model.fit(X_train, y_train)
preds = model.predict(X_train)

# =========================================
# Metrics
# =========================================
R2 = r2_score(y_train, preds)
MAE = mean_absolute_error(y_train, preds)
errors = preds - y_train
STD = np.std(errors)

# =========================================
# Plot
# =========================================
sns.set(style="whitegrid", font_scale=1.3)
plt.figure(figsize=(14, 6))

# -------- LEFT: Error Distribution (KDE curve) --------
plt.subplot(1, 2, 1)
sns.kdeplot(
    errors,
    fill=True,
    linewidth=2.5,
    color="#4C72B0"
)
plt.axvline(0, color="black", linestyle="--", linewidth=1.8)
plt.xlabel("Predicted − Experimental CCS (Å²)")
plt.ylabel("Density")
plt.title(f"Error Distribution\nSTD = {STD:.2f}")
plt.grid(True, linestyle="--", alpha=0.4)

# -------- RIGHT: Predicted vs Experimental --------
plt.subplot(1, 2, 2)

charge_colors = {
    1: "#1f77b4",
    2: "#d62728",
    3: "#9467bd",
    4: "#17becf",
}

for c in sorted(np.unique(charge_train)):
    idx = charge_train == c
    plt.scatter(
        y_train[idx],
        preds[idx],
        s=28,
        color=charge_colors.get(c, "gray"),
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85
    )

# Diagonal line
lims = [y_train.min(), y_train.max()]
plt.plot(lims, lims, "r--", linewidth=2)

# Fitting line
coef = np.polyfit(y_train, preds, 1)
x_fit = np.linspace(*lims, 200)
y_fit = coef[0] * x_fit + coef[1]
plt.plot(x_fit, y_fit, color="blue", linewidth=2)

plt.xlabel("Experimental CCS (Å²)")
plt.ylabel("Predicted CCS (Å²)")
#plt.title(f"R² = {R2:.3f}, MAE = {MAE:.3f} Å²")
plt.title(f"Predicted vs True CCS\nR² = {R2:.3f}, MAE = {MAE:.3f} Å²")
plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()
