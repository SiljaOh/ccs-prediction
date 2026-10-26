"""Figures: data distributions, predicted vs. experimental CCS, error distribution, feature importance."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHARGE_COLORS = {1: "#1f77b4", 2: "#d62728", 3: "#e377c2", 4: "#17becf"}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def ccs_distribution(ccs, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ccs, bins=60, color="#1f77b4")
    ax.axvline(np.median(ccs), color="black", ls="--", lw=1, label=f"median {np.median(ccs):.1f} Å²")
    ax.set(xlabel="CCS (Å²)", ylabel="Count", title="CCS distribution")
    ax.legend(frameon=False)
    _save(fig, path)


def length_distribution(lengths, limit, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(lengths.min(), lengths.max() + 2) - 0.5
    ax.hist(lengths, bins=bins, color="#4C72B0", edgecolor="black", lw=0.4)
    ax.axvline(limit, color="#C2410C", ls="--", label=f"IQR upper limit ({limit:.0f} aa)")
    ax.set(xlabel="Peptide length", ylabel="Count", title="Peptide length distribution")
    ax.legend(frameon=False)
    _save(fig, path)


def predicted_vs_experimental(y_true, y_pred, charge, metrics, title, path, max_points=40000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y_true))[:max_points]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for z in sorted(np.unique(charge)):
        m = idx[charge[idx] == z]
        ax.scatter(y_true[m], y_pred[m], s=10, alpha=0.6, color=CHARGE_COLORS.get(int(z), "gray"),
                   edgecolor="black", linewidth=0.2, label=f"Charge +{int(z)}")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Diagonal line")
    k, d = np.polyfit(y_true, y_pred, 1)
    ax.plot(lims, [k * lims[0] + d, k * lims[1] + d], color="blue", lw=1.5, label="Fitting line")
    ax.set(xlabel="Experimental CCS (Å²)", ylabel="Predicted CCS (Å²)",
           title=f"{title}\nR² = {metrics['r2']:.3f}, MAE = {metrics['mae']:.2f} Å²")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(frameon=True, loc="upper left")
    _save(fig, path)


def error_distribution(y_true, y_pred, title, path):
    err = np.asarray(y_pred) - np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err, bins=np.linspace(-150, 150, 61), color="#4C72B0", edgecolor="black", lw=0.3)
    ax.axvline(0, color="red", ls="--", lw=1, label="Zero error")
    ax.axvline(err.mean(), color="orange", ls="-.", lw=1, label=f"Mean bias ({err.mean():.2f})")
    ax.set(xlabel="Error (predicted − experimental) [Å²]", ylabel="Frequency",
           title=f"{title}\nMAE = {np.abs(err).mean():.2f} Å², σ = {err.std():.2f} Å²")
    ax.legend(frameon=False)
    _save(fig, path)


def feature_importance(groups: dict[str, float], title, path):
    names, vals = list(groups)[::-1], list(groups.values())[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names, vals, color="#5470C6")
    ax.set(xlabel="Normalised importance", title=title)
    _save(fig, path)


def model_comparison(results: dict[str, dict], path):
    names = list(results)
    r2 = [results[n]["r2"] for n in names]
    mae = [results[n]["mae"] for n in names]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.bar(names, r2, color="#1F4E9A")
    a.set(ylim=(min(r2) - 0.01, 1.0), ylabel="R² (test set)", title="R² (higher is better)")
    b.bar(names, mae, color="#C2410C")
    b.set(ylabel="MAE (Å²)", title="MAE (lower is better)")
    for ax, vals, fmt in ((a, r2, "{:.3f}"), (b, mae, "{:.2f}")):
        for i, v in enumerate(vals):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom")
    _save(fig, path)


def loss_curves(history, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="validation")
    ax.set(xlabel="Epoch", ylabel="MSE (normalised CCS)", yscale="log", title=title)
    ax.legend(frameon=False)
    _save(fig, path)
