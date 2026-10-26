"""Feature engineering: one-hot sequences, composition, physicochemical and modification features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import MAX_LENGTH, modifications

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
HYDROPHOBIC = set("AVILMFWC")
MODS = ["ox", "ac"]  # methionine oxidation, protein N-terminal acetylation

# monoisotopic residue masses (Da)
RESIDUE_MASS = {
    "A": 71.03711, "C": 103.00919, "D": 115.02694, "E": 129.04259, "F": 147.06841,
    "G": 57.02146, "H": 137.05891, "I": 113.08406, "K": 128.09496, "L": 113.08406,
    "M": 131.04049, "N": 114.04293, "P": 97.05276, "Q": 128.05858, "R": 156.10111,
    "S": 87.03203, "T": 101.04768, "V": 99.06841, "W": 186.07931, "Y": 163.06333,
}
MOD_MASS = {"ox": 15.99491, "ac": 42.01057}
WATER = 18.01056
PROTON = 1.00728


def one_hot(seq: str, max_length: int = MAX_LENGTH) -> np.ndarray:
    """Sequence -> (max_length, 20) binary matrix, zero padded at the end."""
    x = np.zeros((max_length, len(AMINO_ACIDS)), dtype=np.float32)
    for i, aa in enumerate(seq[:max_length]):
        j = AA_INDEX.get(aa)
        if j is not None:
            x[i, j] = 1.0
    return x


def aa_counts(seq: str) -> np.ndarray:
    return np.array([seq.count(aa) for aa in AMINO_ACIDS], dtype=np.float32)


def count_hydrophobic(seq: str) -> int:
    return sum(aa in HYDROPHOBIC for aa in seq)


def peptide_mass(seq: str, mods: list[str]) -> float:
    return sum(RESIDUE_MASS.get(aa, 0.0) for aa in seq) + WATER + sum(MOD_MASS.get(m, 0.0) for m in mods)


def mz(mass: float, charge: int) -> float:
    return (mass + charge * PROTON) / charge


def modification_flags(mods: list[str]) -> np.ndarray:
    flags = [float(m in mods) for m in MODS]
    return np.array(flags + [float(bool(mods))], dtype=np.float32)  # + isModified


def tabular(df: pd.DataFrame, with_mods: bool = False) -> tuple[np.ndarray, list[str]]:
    """Flat feature matrix for Ridge Regression and Random Forest (up to 690 features)."""
    rows, names = [], None
    for seq, mseq, z in zip(df["Sequence"], df["Modified sequence"], df["Charge"]):
        mods = modifications(mseq)
        mass = peptide_mass(seq, mods)
        parts = [
            np.array([mass, z, mz(mass, z), len(seq), count_hydrophobic(seq)], dtype=np.float32),
            aa_counts(seq),
            one_hot(seq).ravel(),
        ]
        if with_mods:
            parts.append(modification_flags(mods))
        rows.append(np.concatenate(parts))
    names = (["total_mass", "charge", "mz", "length", "count_hydrophobic"]
             + [f"count_{a}" for a in AMINO_ACIDS]
             + [f"ohe_{i}_{a}" for i in range(MAX_LENGTH) for a in AMINO_ACIDS]
             + (["is_oxidated", "is_acetylated", "is_modified"] if with_mods else []))
    return np.vstack(rows), names


def feature_group(name: str) -> str:
    """Map a feature name to the groups used in the feature importance plots."""
    if name.startswith("ohe_"):
        return "OHE sequence"
    if name.startswith("count_") and name != "count_hydrophobic":
        return "count_AA"
    return {"total_mass": "Total mass", "charge": "Charge", "mz": "m/z", "length": "Sequence length",
            "count_hydrophobic": "count_hydrophobic"}.get(name, name)


def sequence_tensors(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inputs for the deep learning models.

    Returns
    -------
    seq : (n, 32, 20) one-hot sequences
    charge : (n, 4) one-hot charge state (+1 .. +4)
    mods : (n, 3) modification flags
    """
    seq = np.stack([one_hot(s) for s in df["Sequence"]])
    charge = np.zeros((len(df), 4), dtype=np.float32)
    charge[np.arange(len(df)), np.clip(df["Charge"].to_numpy() - 1, 0, 3)] = 1.0
    mods = np.stack([modification_flags(modifications(m)) for m in df["Modified sequence"]])
    return seq, charge, mods
