#!/usr/bin/env python3
"""
multi_sample_simulation.py

Simulate multi–sample (M-dimensional) tumor sequencing data with known ground truth,
covering a grid of parameters: tumor purity, CNA rate, read depth, and number of clusters.
For each combination, generates multiple replicates and saves:
  - observations:    {PREFIX}_purity{rho}_cna{p}_depth{N}_K{K}_{rep}_obs.csv
  - true parameters: {PREFIX}_purity{rho}_cna{p}_depth{N}_K{K}_{rep}_params.pkl
  - true clusters:   {PREFIX}_purity{rho}_cna{p}_depth{N}_K{K}_{rep}_clusters.csv

Based on the CliPPSim4k design (Extended Data Fig.2a, Methods):
  • Purity ρ ∈ {0.3, 0.6, 0.9}
  • CNA rate p ∈ {0.0, 0.1, 0.2}
  • Depth N ∈ {100, 500, 1000}
  • Number of clusters K ∈ {2, 3, 4}
  • SNV count per cluster ∼ Uniform[200, 500]
  • Cluster CCF β_k ∼ Uniform[0.2, 1.0], min pairwise distance d=0.2
  • Cellular prevalence ϕ_k = ρ * β_k

Each combination yields `REPS = 50` datasets.

Usage:
  python multi_sample_simulation.py
"""
import os
import pickle
import numpy as np
import itertools as its
import pandas as pd
from scipy.stats import bernoulli
from pathlib import Path
# The `clipp2.core` import was not used, so it has been removed.
from typing import Callable, List

# --- Simulation settings ---
M_LIST = [2, 5, 10, 15]            # number of tumor samples per dataset
PURITY_LIST = [0.3, 0.6, 0.9]    # tumor purity
CNA_RATE_LIST = [0.0, 0.1, 0.2]   # CNA rate
DEPTH_LIST = [20, 30, 100]     # sequencing depths
CLUSTER_LIST = [2, 5, 10]         # number of mutation clusters
REPS = 10                        # replicates per combination
PREFIX = 'CliPP2Sim4k'            # filename prefix
OUT_DIR = Path('simulations')
OUT_DIR.mkdir(exist_ok=True)
META_FILE = 'metaData.pkl'

# --- Linear approximation utilities ---
def theta(w, bv, cv, cn):
    # Added a small epsilon to the denominator to prevent division by zero if cn is 0
    return (np.exp(w) * bv) / (cn + np.exp(w) * cv + 1e-9)

def LinearEvaluate(x, a, b):
    return a * x + b

def LinearApproximate(bv, cv, cn):
    w = np.arange(-40, 41) / 10.0
    actual_theta = theta(w, bv, cv, cn)
    No_w = len(w)
    n_pairs = int((No_w - 3) * (No_w - 2) / 2)
    diffs = np.zeros(n_pairs)
    coefs = np.zeros((n_pairs, 6))
    cuts = np.zeros((n_pairs, 2))
    k = 0
    for i in range(1, No_w - 2):
        for j in range(i + 1, No_w - 1):
            cuts[k] = [w[i], w[j]]
            coefs[k,0] = (actual_theta[i] - actual_theta[0]) / (w[i] - w[0])
            coefs[k,1] = actual_theta[0] - w[0] * coefs[k,0]
            coefs[k,2] = (actual_theta[j] - actual_theta[i]) / (w[j] - w[i])
            coefs[k,3] = actual_theta[i] - w[i] * coefs[k,2]
            coefs[k,4] = (actual_theta[-1] - actual_theta[j]) / (w[-1] - w[j])
            coefs[k,5] = actual_theta[-1] - w[-1] * coefs[k,4]
            seg1 = LinearEvaluate(w[:i+1], coefs[k,0], coefs[k,1])
            seg2 = LinearEvaluate(w[i+1:j+1], coefs[k,2], coefs[k,3])
            seg3 = LinearEvaluate(w[j+1:], coefs[k,4], coefs[k,5])
            approx = np.concatenate([seg1, seg2, seg3])
            diffs[k] = np.max(np.abs(actual_theta - approx))
            k += 1
    K = np.argmin(diffs)
    return {'w_cut': cuts[K], 'diff': diffs[K], 'coef': coefs[K]}

class DirichletProcessSampler:
    """
    Generates independent and identically distributed (i.i.d.) integers
    by first drawing a discrete distribution P from a Dirichlet Process DP(alpha, H),
    and then drawing samples from P.
    """

    def __init__(self, alpha: float, base_distribution: Callable[[int], np.ndarray], truncation_level: int = 500):
        if alpha <= 0:
            raise ValueError("Alpha must be positive.")
        if truncation_level <= 0:
            raise ValueError("Truncation level must be positive.")
            
        self.alpha = alpha
        self.base_distribution = base_distribution
        self.truncation_level = truncation_level
        betas = np.random.beta(1, self.alpha, self.truncation_level)
        remaining_stick = np.cumprod(1 - betas)
        weights = betas * np.insert(remaining_stick[:-1], 0, 1)
        weights /= np.sum(weights)
        atoms = self.base_distribution(self.truncation_level)
        self.atoms = atoms
        self.weights = weights

    def draw(self) -> int:
        return np.random.choice(self.atoms, p=self.weights)

    def draw_many(self, size: int) -> np.ndarray:
        return np.random.choice(self.atoms, p=self.weights, size=size, replace=True)

# --- Load or init metadata cache ---
if Path(META_FILE).exists():
    with open(META_FILE, 'rb') as f:
        metaData = pickle.load(f)
else:
    metaData = {}

# --- Main simulation loops ---
param_combinations = list(its.product(PURITY_LIST, CNA_RATE_LIST, DEPTH_LIST, CLUSTER_LIST, M_LIST))
total_sims = len(param_combinations) * REPS

print(f"Starting simulation for {len(param_combinations)} parameter combinations, {REPS} reps each...")
count = 0

for rho, p_cna, N, K, M in param_combinations:
    for rep in range(REPS):
        # 1) cluster sizes and mutation count
        base_dist_func = lambda size: np.random.randint(200, 500, size=size)
        dp_sampler_high_alpha = DirichletProcessSampler(alpha=50.0, base_distribution=base_dist_func)
        cluster_size = dp_sampler_high_alpha.draw_many(K)
        No_mutations = sum(cluster_size)

        # 2) sample cluster-level CCFs
        cluster_ccf = np.random.uniform(0, 1.0, size=(K - 1, M))
        # FIX 1: Replaced deprecated `np.row_stack` with `np.vstack`
        cluster_ccf = np.vstack((np.ones(M), cluster_ccf))
        
        # enforce min Euclidean distance d=0.2
        dists = [np.linalg.norm(cluster_ccf[i] - cluster_ccf[j])
                 for i, j in its.combinations(range(K), 2)]
        while min(dists) < 0.2 or np.any(np.linalg.norm(cluster_ccf, axis=1) == 0):
            cluster_ccf = np.random.uniform(0, 1.0, size=(K - 1, M))
            cluster_ccf = np.vstack((np.ones(M), cluster_ccf))
            dists = [np.linalg.norm(cluster_ccf[i] - cluster_ccf[j])
                     for i, j in its.combinations(range(K), 2)]

        # 3) per-mutation CCF and cellular prevalence
        mutation_ccf = np.vstack([
            np.tile(cluster_ccf[k], (cluster_size[k], 1)) for k in range(K)
        ])
        mutation_cp = rho * mutation_ccf

        # 4) initialize copy numbers (mutated/ref)
        minor = np.ones((No_mutations, M), dtype=int)
        total = np.ones((No_mutations, M), dtype=int) * 2

        # 5) sample CNA indicators
        mask = bernoulli.rvs(p_cna, size=(No_mutations, M))
        if p_cna > 0:
            mut_amp = np.random.randint(0, 6, size=(No_mutations, M))
            ref_amp = np.random.randint(0, 6, size=(No_mutations, M))
            
            # Avoid total copy number of 0, which is biologically non-viable for this simulation
            # and causes division by zero later.
            total_cna = mut_amp + ref_amp
            total_cna[total_cna == 0] = 1 # Set homozygous deletions to copy number 1

            minor[mask==1] = mut_amp[mask==1]
            total[mask==1] = total_cna[mask==1]

        # 6) simulate read counts
        N_mut = (total / 2) * N
        n_obs = np.random.poisson(N_mut)
        # Add a small epsilon to the vaf denominator to prevent 0/0 when total=0
        vaf = (mutation_cp * minor) / (2*(1 - rho) + rho * total + 1e-9)
        # Ensure VAF is clipped between 0 and 1
        vaf = np.clip(vaf, 0, 1)
        r_obs = np.random.binomial(n_obs, vaf)

        # 7) estimate minor from data
        # FIX 2: Prevent division by zero when n_obs is 0.
        # Initialize est with zeros.
        est = np.zeros_like(r_obs, dtype=float)
        # Create a mask for safe division (where n_obs > 0).
        safe_div_mask = n_obs > 0
        
        # Calculate the numerator and denominator for the safe entries.
        numerator = r_obs[safe_div_mask] * (rho * total[safe_div_mask] + 2 * (1 - rho))
        denominator = n_obs[safe_div_mask] * rho
        
        # Perform the division only on the safe entries.
        est[safe_div_mask] = np.round(numerator / denominator)
        
        # The rest of the logic can now proceed without NaN values.
        est = np.minimum.reduce([est, np.maximum.reduce([minor, total - minor])])
        minor_est = est.astype(int)
        minor_est[minor_est == 0] = 1

        # 8) get phasing coefficients
        coef_arr = np.zeros((No_mutations, M, 6))
        cut_arr  = np.zeros((No_mutations, M, 2))
        for i in range(No_mutations):
            for j in range(M):
                key = f"2_{minor_est[i,j]}_{total[i,j]}"
                if key in metaData:
                    coef_arr[i,j] = metaData[key]['coef']
                    cut_arr[i,j]  = metaData[key]['w_cut']
                else:
                    tmp = LinearApproximate(minor_est[i,j], total[i,j], 2)
                    coef_arr[i,j]     = tmp['coef']
                    cut_arr[i,j]      = tmp['w_cut']
                    metaData[key]     = tmp

        # save updated metadata
        with open(META_FILE, 'wb') as f:
            pickle.dump(metaData, f)

        # 9) assemble observation DataFrame
        true_clusters = np.repeat(np.arange(K), cluster_size)
        df = pd.DataFrame({
            'mutation_id': np.repeat(np.arange(No_mutations), M),
            'sample_id':   np.tile(np.arange(M), No_mutations),
            'depth':       n_obs.flatten(),
            'reads':       r_obs.flatten(),
            'minor_true':  minor.flatten(),
            'total_true':  total.flatten(),
            'minor_est':   minor_est.flatten(),
            'cluster_id':  np.repeat(true_clusters, M)
        })

        # 10) save outputs with parameter-coded filenames
        base = f"{PREFIX}_purity{rho}_cna{p_cna}_depth{N}_K{K}_M{M}_{rep}"
        df.to_csv(OUT_DIR/f"{base}_obs.csv", index=False)
        with open(OUT_DIR/f"{base}_params.pkl", 'wb') as f:
            pickle.dump({'rho': rho, 'p_cna': p_cna, 'depth': N,
                         'K': K, 'M' : M, 'cluster_size': cluster_size,
                         'cluster_ccf': cluster_ccf.tolist()}, f)
        np.savetxt(OUT_DIR/f"{base}_clusters.csv", true_clusters,
                   fmt='%d', header='cluster_id', comments='')
        
        count += 1
        if count % 100 == 0:
            print(f"  ... completed {count}/{total_sims} datasets")


print(f"Simulation complete: generated {total_sims} datasets.")