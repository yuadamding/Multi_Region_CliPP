#!/usr/bin/env python3
"""
multi_sample_simulation.py  ––  v2025-07-03
-------------------------------------------
* Generates multi-sample mutation data sets with a tree-structured cluster
  CCF (Cancer Cell Fraction) model.
* Rewritten for speed, reproducibility and NumPy-2.0 compliance.

Output files (same as before):
  <PREFIX>_obs.csv          – per-mutation read counts & CNAs
  <PREFIX>_params.json      – simulation parameters (NumPy-aware encoder)
  <PREFIX>_clusters.csv     – one cluster label per mutation
"""

from __future__ import annotations

import json, itertools as its, math, os
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import bernoulli
from scipy.spatial.distance import pdist

# ---------------------------------------------------------------- Simulation grid
M_LIST          = [2, 5, 10]
PURITY_LIST     = [0.3, 0.6, 0.9]                    # ρ
CNA_RATE_LIST   = [0.0, 0.1, 0.2]                    # Bernoulli(p_cna)
DEPTH_LIST      = [30, 100]                          # sequencing depth per allele
CLUSTER_LIST    = [2, 5, 10]                         # K
SPARSITY_LIST   = [0.2, 0.4, 0.6]                    # per-sample dropout prob.

REPS    = 1
PREFIX  = "CliPP2Sim"
OUT_DIR = Path("simulations_tree")
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- RNG utilities
RNG = np.random.default_rng(seed=20250703)           # global generator

def rng_uniform(low: float, high: float, size=None):
    return RNG.uniform(low, high, size)

def rng_choice(a, size=None, p=None, replace=True):
    return RNG.choice(a, size=size, p=p, replace=replace)

def rng_beta(a, b, size=None):
    return RNG.beta(a, b, size=size)

def rng_binom(n, p, size=None):
    return RNG.binomial(n, p, size=size)

def rng_poisson(lam, size=None):
    return RNG.poisson(lam, size=size)

# ---------------------------------------------------------------- JSON encoder
class NumpyEncoder(json.JSONEncoder):
    """Json encoder for NumPy scalars / arrays (NumPy 2.0 compatible)."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ---------------------------------------------------------------- misc helpers
def theta(w: np.ndarray, bv, cv, cn) -> np.ndarray:
    return (np.exp(w) * bv) / (cn + np.exp(w) * cv + 1e-9)

# ---------------------------------------------------------------- Dirichlet process
class DirichletProcessSampler:
    """Stick-breaking with finite truncation; vectorised for speed."""
    def __init__(self,
                 alpha: float,
                 base_distribution: Callable[[int], np.ndarray],
                 trunc: int = 512) -> None:
        self.trunc = trunc
        betas   = rng_beta(1, alpha, trunc)
        sticks  = np.concatenate(([1.0], np.cumprod(1 - betas[:-1])))
        self.weights = betas * sticks
        self.weights /= self.weights.sum()           # ensure normalisation
        self.atoms   = base_distribution(trunc)

    def draw_many(self, size: int) -> np.ndarray:
        idx = rng_choice(np.arange(self.trunc), size=size, p=self.weights)
        return self.atoms[idx]

# ---------------------------------------------------------------- tree-structured CCF
# ---------------------------------------------------------------- tree-structured CCF
# ---------------------------------------------------------------- tree-structured CCF
def generate_tree_ccfs(K: int, M: int, sparsity: float,
                       max_try: int = 100           # NEW: sibling-split guard
                       ) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Returns a (K, M) CCF matrix and the parent map; guaranteed to terminate.

    Changes (2025-07-04)
    --------------------
    * The sibling split loop is now bounded by `max_try`.  After that many
      unsuccessful draws we fall back to a deterministic (0.3, 0.5) split
      that satisfies all constraints, so no infinite loop is possible.
    """
    if K == 1:
        return np.ones((1, M), dtype=float), {}

    # ---------- 1. random binary tree ------------------------------------
    parent_of: Dict[int, int] = {}
    children = {i: [] for i in range(K)}
    available = [0]
    for node in range(1, K):
        par = rng_choice(available)
        parent_of[node] = par
        children[par].append(node)
        available.append(node)
        if len(children[par]) == 2:
            available.remove(par)

    # ---------- 2. top-down CCF assignment -------------------------------
    ccf = np.zeros((K, M), dtype=float)
    ccf[0] = 1.0                                   # clonal root

    queue = [0]
    while queue:
        par = queue.pop(0)
        p_vec = ccf[par]
        kids  = children[par]
        if not p_vec.any():                        # parent inactive everywhere
            continue
        queue.extend(kids)

        present = rng_uniform(0, 1, (len(kids), M)) > sparsity
        active_idx = np.where(p_vec > 0)[0]
        for kdx, m in enumerate(present):
            if not m.any() and active_idx.size:
                m[rng_choice(active_idx)] = True
            present[kdx] = m

        # ---- sample child fractions (bounded loops) --------------------
        for m in range(M):
            pc = p_vec[m]
            if pc == 0:
                continue
            act = np.flatnonzero(present[:, m])
            if act.size == 0:
                continue

            if act.size == 1:
                ccf[kids[act[0]], m] = pc * rng_uniform(0.1, 0.8)

            elif act.size == 2:
                ok, tries = False, 0
                while not ok and tries < max_try:
                    tries += 1
                    f1 = rng_uniform(0.1, 0.8)
                    f2_max = min(0.8, 1.0 - f1)
                    if f2_max <= 0.1:
                        continue
                    f2 = rng_uniform(0.1, f2_max)
                    ok = abs(f1 - f2) > 0.2
                if not ok:                         # fallback split
                    f1, f2 = 0.3, 0.5
                ccf[kids[act[0]], m] = pc * f1
                ccf[kids[act[1]], m] = pc * f2

            else:                                  # act.size > 2
                fr = rng_uniform(0.1, 0.3, act.size)
                s  = fr.sum() or 1.0               # avoid /0
                fr *= 0.8 / s
                ccf[[kids[a] for a in act], m] = pc * fr

    return ccf, parent_of


# ---------------------------------------------------------------- main loop
def main() -> None:
    param_grid = list(its.product(PURITY_LIST, CNA_RATE_LIST,
                                  DEPTH_LIST, CLUSTER_LIST,
                                  M_LIST, SPARSITY_LIST))
    total = len(param_grid) * REPS
    print(f"Creating {total} data sets ...")

    max_ccf_try = 200          # NEW: limit for distance-check attempts
    base_thr    = 0.20         # initial min-distance requirement

    sim_counter = 0
    for rho, p_cna, depth, K, M, sparsity in param_grid:
        for rep in range(REPS):

            # ---- cluster sizes (unchanged) ------------------------------
            base_dist = lambda sz: RNG.integers(200, 500, size=sz, dtype=np.int64)
            dp  = DirichletProcessSampler(alpha=50.0, base_distribution=base_dist)
            cluster_sz = dp.draw_many(K)
            N_mut = int(cluster_sz.sum())

            # ---- cluster CCFs  (bounded loop) ---------------------------
            thr = base_thr
            for attempt in range(1, max_ccf_try + 1):
                ccf, tree = generate_tree_ccfs(K, M, sparsity)
                if K == 1 or pdist(ccf).min(initial=np.inf) >= thr:
                    break
                # every 50 attempts soften the criterion slightly
                if attempt % 50 == 0:
                    thr *= 0.9                      # e.g. 0.20 → 0.18 → 0.162 ...
            else:
                # last resort: accept whatever we have after max_ccf_try
                print(f"[WARN] Could not reach min-distance {base_thr:.2f} "
                      f"after {max_ccf_try} tries (K={K}, M={M}, sparsity={sparsity}); "
                      "using relaxed CCFs.")

            # ---------- per-mutation CCF & CP ----------------------------
            mut_ccf = np.repeat(ccf, cluster_sz, axis=0)
            mut_cp  = rho * mut_ccf

            # ---------- CNA generation -----------------------------------
            minor = np.ones((N_mut, M), np.int64)
            total = np.full((N_mut, M), 2, np.int64)
            if p_cna > 0:
                mask = bernoulli.rvs(p_cna, size=(N_mut, M), random_state=RNG)
                mut_amp = RNG.integers(0, 6, size=(N_mut, M))
                ref_amp = RNG.integers(0, 6, size=(N_mut, M))
                total_cna = (mut_amp + ref_amp).clip(min=1)
                minor[mask == 1] = mut_amp[mask == 1]
                total[mask == 1] = total_cna[mask == 1]

            # ---------- depth / reads ------------------------------------
            n_exp = (total / 2) * depth
            n_obs = rng_poisson(n_exp)

            vaf   = (mut_cp * minor) / (2 * (1 - rho) + rho * total + 1e-9)
            vaf   = vaf.clip(0, 1)
            r_obs = rng_binom(n_obs, vaf)

            # ---------- ploidy estimates ---------------------------------
            est = np.zeros_like(r_obs, float)
            mask = n_obs > 0
            num = r_obs[mask] * (rho * total[mask] + 2 * (1 - rho))
            est[mask] = np.round(num / (n_obs[mask] * rho + 1e-9))
            est = np.minimum(est, np.maximum(minor, total - minor))
            minor_est = est.clip(min=1).astype(int)

            # ---------- cluster labels -----------------------------------
            labels = np.repeat(np.arange(K, dtype=np.int32), cluster_sz)

            # ---------- write outputs ------------------------------------
            tag = (f"{PREFIX}_purity{rho}_cna{p_cna}_depth{depth}"
                   f"_K{K}_M{M}_sparse{sparsity}_{rep}")
            df_obs = pd.DataFrame({
                "mutation_id": np.repeat(np.arange(N_mut, dtype=np.int32), M),
                "sample_id":   np.tile(np.arange(M, dtype=np.int32), N_mut),
                "depth":       n_obs.flatten(),
                "reads":       r_obs.flatten(),
                "minor_true":  minor.flatten(),
                "total_true":  total.flatten(),
                "minor_est":   minor_est.flatten(),
                "cluster_id":  np.repeat(labels, M)
            })
            df_obs.to_csv(OUT_DIR / f"{tag}_obs.csv", index=False)

            with open(OUT_DIR / f"{tag}_params.json", "w", encoding="utf-8") as fh:
                json.dump({
                    "rho": rho, "p_cna": p_cna, "depth": depth,
                    "K": K, "M": M, "sparsity_rate": sparsity,
                    "cluster_size": cluster_sz,
                    "cluster_ccf": ccf,
                    "tree_structure": tree
                }, fh, cls=NumpyEncoder, indent=2)

            np.savetxt(OUT_DIR / f"{tag}_clusters.csv",
                       labels, fmt="%d", header="cluster_id", comments="")

            sim_counter += 1
            if sim_counter % 50 == 0:
                print(f"  ... {sim_counter}/{total} completed")

    print(f"Finished: {total} data sets written to '{OUT_DIR}'")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
