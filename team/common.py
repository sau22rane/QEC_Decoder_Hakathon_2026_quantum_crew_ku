"""Shared helpers for the team: circuits, problem matrices, fast data loading and generation.

Data is cached as bit-packed .npz files in data/generated/ (git-ignored), which load in milliseconds
instead of the ~seconds np.loadtxt needs for the CSVs.
"""
import os
from functools import lru_cache

import numpy as np
import stim

from hackathon_2026 import apply_circuit_noise, create_circuit, load_kaggle_csv, problem_info

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_DIR = os.path.join(ROOT, "data", "generated")
NUM_DETECTORS, NUM_OBSERVABLES = 70, 4
P_COMPETITION, ROUNDS = 0.002, 5


@lru_cache(maxsize=None)
def circuit(basis: str, p: float = P_COMPETITION, rounds: int = ROUNDS) -> stim.Circuit:
    """Noisy circuit for one basis. With the defaults it equals data/train/<basis>_circuit.stim."""
    return apply_circuit_noise(create_circuit(basis, rounds), p=p)


@lru_cache(maxsize=None)
def matrices(basis: str, p: float = P_COMPETITION):
    """(H, L, priors) for one basis: H is (70, n_mech), L is (4, n_mech), priors is (n_mech,)."""
    info = problem_info(circuit(basis, p))
    return info["H"], info["L"], info["p"]


def generate(basis: str, shots: int, seed: int, p: float = P_COMPETITION, cache: bool = True):
    """Sample labelled shots (detectors bool (shots, 70), observables bool (shots, 4)), cached on disk."""
    path = os.path.join(GEN_DIR, f"{basis}_p{p}_n{shots}_s{seed}.npz")
    if cache and os.path.exists(path):
        f = np.load(path)
        return (np.unpackbits(f["d"], axis=1, count=NUM_DETECTORS).astype(bool),
                np.unpackbits(f["o"], axis=1, count=NUM_OBSERVABLES).astype(bool))
    d, o = circuit(basis, p).compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    if cache:
        os.makedirs(GEN_DIR, exist_ok=True)
        np.savez_compressed(path, d=np.packbits(d, axis=1), o=np.packbits(o, axis=1))
    return d, o


@lru_cache(maxsize=None)
def labelled_set(name: str = "train"):
    """The labelled CSV that ships with the repo (data/train), split by basis: {basis: (detectors, observables)}."""
    data = load_kaggle_csv(os.path.join(ROOT, "data", name, f"{name}.csv"))
    return {b: (data["detector_events"][data["basis"] == b], data["observable_flips"][data["basis"] == b]) for b in "XZ"}


def ler(pred: np.ndarray, true: np.ndarray) -> float:
    """Leaderboard metric: fraction of wrong observable bits."""
    return float(np.mean(np.asarray(pred).astype(bool) != np.asarray(true).astype(bool)))
