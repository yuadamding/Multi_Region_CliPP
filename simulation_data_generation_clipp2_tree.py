#!/usr/bin/env python3
"""
multi_sample_simulation.py

UPDATED: Generates cluster Cancer Cell Fractions (CCFs) based on a random
binary tree structure, enforcing the sum-of-CCFs constraint to respect
the Pigeonhole Principle and Infinite Sites Assumption for subclones.
... (rest of the docstring) ...
"""
import os
import pickle
import numpy as np
import itertools as its
import pandas as pd
from scipy.stats import bernoulli
from pathlib import Path
from typing import Callable, List, Dict

# --- Simulation settings ---
M_LIST = [2, 5, 10, 15]
PURITY_LIST = [0.3, 0.6, 0.9]
CNA_RATE_LIST = [0.0, 0.1, 0.2]
DEPTH_LIST = [20, 30, 100]
CLUSTER_LIST = [2, 5, 10]
REPS = 10
PREFIX = 'CliPP2Sim4k_Tree' # Updated prefix for new simulation files
OUT_DIR = Path('simulations_tree')
OUT_DIR.mkdir(exist_ok=True)
META_FILE = 'metaData_tree.pkl'

# --- Linear approximation utilities (Unchanged) ---
def theta(w, bv, cv, cn):
    return (np.exp(w) * bv) / (cn + np.exp(w) * cv + 1e-9)
def LinearEvaluate(x, a, b):
    return a * x + b
def LinearApproximate(bv, cv, cn):
    w, No_w = np.arange(-40, 41) / 10.0, 81
    actual_theta = theta(w, bv, cv, cn)
    n_pairs = int((No_w - 3) * (No_w - 2) / 2)
    diffs, coefs, cuts = np.zeros(n_pairs), np.zeros((n_pairs, 6)), np.zeros((n_pairs, 2))
    k = 0
    for i in range(1, No_w - 2):
        for j in range(i + 1, No_w - 1):
            cuts[k] = [w[i], w[j]]
            coefs[k,0] = (actual_theta[i] - actual_theta[0]) / (w[i] - w[0]); coefs[k,1] = actual_theta[0] - w[0] * coefs[k,0]
            coefs[k,2] = (actual_theta[j] - actual_theta[i]) / (w[j] - w[i]); coefs[k,3] = actual_theta[i] - w[i] * coefs[k,2]
            coefs[k,4] = (actual_theta[-1] - actual_theta[j]) / (w[-1] - w[j]); coefs[k,5] = actual_theta[-1] - w[-1] * coefs[k,4]
            seg1 = LinearEvaluate(w[:i+1], coefs[k,0], coefs[k,1]); seg2 = LinearEvaluate(w[i+1:j+1], coefs[k,2], coefs[k,3]); seg3 = LinearEvaluate(w[j+1:], coefs[k,4], coefs[k,5])
            diffs[k] = np.max(np.abs(actual_theta - np.concatenate([seg1, seg2, seg3])))
            k += 1
    K_min = np.argmin(diffs)
    return {'w_cut': cuts[K_min], 'diff': diffs[K_min], 'coef': coefs[K_min]}

# --- Dirichlet Process for Cluster Sizes (Unchanged) ---
class DirichletProcessSampler:
    def __init__(self, alpha: float, base_distribution: Callable[[int], np.ndarray], truncation_level: int = 500):
        if alpha <= 0: raise ValueError("Alpha must be positive.")
        if truncation_level <= 0: raise ValueError("Truncation level must be positive.")
        self.alpha, self.base_distribution, self.truncation_level = alpha, base_distribution, truncation_level
        betas = np.random.beta(1, self.alpha, self.truncation_level)
        remaining_stick = np.cumprod(1 - betas)
        weights = betas * np.insert(remaining_stick[:-1], 0, 1)
        self.weights = weights / np.sum(weights)
        self.atoms = self.base_distribution(self.truncation_level)
    def draw_many(self, size: int) -> np.ndarray:
        return np.random.choice(self.atoms, p=self.weights, size=size, replace=True)

# ==============================================================================
# NEW: Tree-Structured CCF Generation
# ==============================================================================
def generate_tree_structured_ccfs(K: int, M: int) -> (np.ndarray, Dict):
    """
    Generates a (K, M) CCF matrix respecting a random binary tree structure.

    Args:
        K: The total number of clusters (nodes in the tree).
        M: The number of samples.

    Returns:
        A tuple containing:
        - ccf_matrix (np.ndarray): The (K, M) matrix of generated CCFs.
        - tree (Dict): A dictionary representing the parent of each node.
    """
    if K < 1:
        return np.array([]), {}

    # 1. Build the random binary tree topology
    parent_of = {}
    children_of = {i: [] for i in range(K)}
    available_parents = [0] # Start with the root node

    for i in range(1, K):
        # Pick a random available parent
        parent_idx = np.random.choice(available_parents)
        parent_of[i] = parent_idx
        children_of[parent_idx].append(i)
        
        # The new node is now also an available parent
        available_parents.append(i)
        
        # If the chosen parent is now full (has 2 children), remove it
        if len(children_of[parent_idx]) == 2:
            available_parents.remove(parent_idx)

    # 2. Generate CCFs top-down following the tree structure
    ccf_matrix = np.zeros((K, M))
    
    # The root (clonal) cluster always has a CCF of 1.0
    ccf_matrix[0, :] = 1.0

    # Use a queue for breadth-first traversal to ensure parents are processed before children
    queue = [0] 
    head = 0
    while head < len(queue):
        parent_idx = queue[head]
        head += 1
        
        parent_ccf = ccf_matrix[parent_idx, :]
        children = children_of[parent_idx]
        
        if not children:
            continue

        # Partition the parent's CCF among its children for each sample
        # This enforces the sum-of-CCFs rule: sum(child_ccf) <= parent_ccf
        if len(children) == 1:
            child_idx = children[0]
            # Child CCF is a random fraction of the parent's CCF
            ccf_matrix[child_idx, :] = parent_ccf * np.random.uniform(0.1, 0.9, size=M)
        else: # Two children
            child1_idx, child2_idx = children[0], children[1]
            # Generate two random fractions that sum to < 1
            split1 = np.random.uniform(0.1, 0.8, size=M)
            split2 = np.random.uniform(0.1, 1.0 - split1, size=M)
            
            ccf_matrix[child1_idx, :] = parent_ccf * split1
            ccf_matrix[child2_idx, :] = parent_ccf * split2
            
        queue.extend(children)
        
    return ccf_matrix, parent_of

# --- Load or init metadata cache ---
if Path(META_FILE).exists():
    with open(META_FILE, 'rb') as f: metaData = pickle.load(f)
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
        dp_sampler = DirichletProcessSampler(alpha=50.0, base_distribution=base_dist_func)
        cluster_size = dp_sampler.draw_many(K)
        No_mutations = sum(cluster_size)

        # 2) sample cluster-level CCFs using the new tree-based method
        while True:
            cluster_ccf, tree_structure = generate_tree_structured_ccfs(K, M)
            # Optional: enforce min distance between any two clusters for better separation
            dists = [np.linalg.norm(cluster_ccf[i] - cluster_ccf[j]) for i, j in its.combinations(range(K), 2)]
            if not dists or min(dists) >= 0.1: # Use a slightly more relaxed distance for tree
                break
        
        # 3) per-mutation CCF and cellular prevalence
        mutation_ccf = np.vstack([np.tile(cluster_ccf[k], (cluster_size[k], 1)) for k in range(K)])
        mutation_cp = rho * mutation_ccf

        # Steps 4-10 remain the same, as they operate on the generated CCF matrix
        # 4) initialize copy numbers
        minor = np.ones((No_mutations, M), dtype=int)
        total = np.full((No_mutations, M), 2, dtype=int)

        # 5) sample CNA indicators
        if p_cna > 0:
            mask = bernoulli.rvs(p_cna, size=(No_mutations, M))
            mut_amp = np.random.randint(0, 6, size=(No_mutations, M))
            ref_amp = np.random.randint(0, 6, size=(No_mutations, M))
            total_cna = mut_amp + ref_amp
            total_cna[total_cna == 0] = 1
            minor[mask==1] = mut_amp[mask==1]
            total[mask==1] = total_cna[mask==1]

        # 6) simulate read counts
        N_mut = (total / 2) * N
        n_obs = np.random.poisson(N_mut)
        vaf = np.clip((mutation_cp * minor) / (2*(1 - rho) + rho * total + 1e-9), 0, 1)
        r_obs = np.random.binomial(n_obs, vaf)

        # 7) estimate minor from data
        est = np.zeros_like(r_obs, dtype=float)
        safe_div_mask = n_obs > 0
        numerator = r_obs[safe_div_mask] * (rho * total[safe_div_mask] + 2 * (1 - rho))
        denominator = n_obs[safe_div_mask] * rho
        est[safe_div_mask] = np.round(numerator / (denominator + 1e-9))
        est = np.minimum(est, np.maximum(minor, total - minor))
        minor_est = np.maximum(est.astype(int), 1)

        # 8) get phasing coefficients
        coef_arr, cut_arr  = np.zeros((No_mutations, M, 6)), np.zeros((No_mutations, M, 2))
        for i, j in its.product(range(No_mutations), range(M)):
            key = f"2_{minor_est[i,j]}_{total[i,j]}"
            if key not in metaData:
                metaData[key] = LinearApproximate(minor_est[i,j], total[i,j], 2)
            coef_arr[i,j], cut_arr[i,j] = metaData[key]['coef'], metaData[key]['w_cut']
        with open(META_FILE, 'wb') as f: pickle.dump(metaData, f)

        # 9) assemble observation DataFrame
        true_clusters = np.repeat(np.arange(K), cluster_size)
        df = pd.DataFrame({'mutation_id': np.repeat(np.arange(No_mutations), M),'sample_id': np.tile(np.arange(M), No_mutations),'depth': n_obs.flatten(),'reads': r_obs.flatten(),'minor_true': minor.flatten(),'total_true': total.flatten(),'minor_est': minor_est.flatten(),'cluster_id': np.repeat(true_clusters, M)})

        # 10) save outputs
        base = f"{PREFIX}_purity{rho}_cna{p_cna}_depth{N}_K{K}_M{M}_{rep}"
        df.to_csv(OUT_DIR/f"{base}_obs.csv", index=False)
        with open(OUT_DIR/f"{base}_params.pkl", 'wb') as f:
            pickle.dump({'rho': rho, 'p_cna': p_cna, 'depth': N, 'K': K, 'M' : M, 
                         'cluster_size': cluster_size, 'cluster_ccf': cluster_ccf.tolist(),
                         'tree_structure': tree_structure}, f)
        np.savetxt(OUT_DIR/f"{base}_clusters.csv", true_clusters, fmt='%d', header='cluster_id', comments='')
        
        count += 1
        if count % 100 == 0:
            print(f"  ... completed {count}/{total_sims} datasets")

print(f"Simulation complete: generated {total_sims} datasets into '{OUT_DIR}' directory.")