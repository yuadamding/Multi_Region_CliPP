from __future__ import annotations
import glob, json, os, re, sys
from math import log, sqrt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.metrics import adjusted_rand_score

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
USE_DEPTH_WEIGHT   = False
SCALE_LAMBDA       = 10.0
EXPORT_BIG_ARRAYS  = False

LEAST_MUT_FRAC     = 0.05    # fraction of N; <1 ⇒ relative, ≥1 ⇒ absolute
LEAST_DIFF_RAW     = 0.02    # raw per‑sample gap, scaled by √m later
MAX_POST_MERGES    = 100    # safety to avoid infinite loop

# ──────────────────────────────────────────────────────────────────────
# λ parser
# ──────────────────────────────────────────────────────────────────────
_LAMBDA_RE = re.compile(r'lambda_([0-9eE.+-]+)\.tsv$')
def _parse_lambda(fname: str) -> float:
    m = _LAMBDA_RE.search(fname)
    if not m:
        raise ValueError(f"Cannot parse λ from {fname}")
    lam = float(m.group(1))
    if lam < 0:
        raise ValueError("λ must be non‑negative")
    return SCALE_LAMBDA * lam

# ──────────────────────────────────────────────────────────────────────
# Post‑processing helpers
# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# Helper: recompute centroids & counts
# ──────────────────────────────────────────────────────────────────────
def _reindex(phi: np.ndarray, labels: np.ndarray):
    """
    Return (labels0‑based, centroids, counts, uniq_labels)
    """
    uniq, inv = np.unique(labels, return_inverse=True)
    k, m = len(uniq), phi.shape[1]
    cent = np.zeros((k, m), dtype=phi.dtype)
    cnt  = np.zeros(k,    dtype=np.int64)
    for i, u in enumerate(uniq):
        mask = inv == i
        cent[i] = phi[mask].mean(0)
        cnt[i]  = mask.sum()
    return inv, cent, cnt, uniq


# ──────────────────────────────────────────────────────────────────────
#  A. Dissolve ONE undersized cluster (< least_mut)
# ──────────────────────────────────────────────────────────────────────
def _merge_one_small(phi, labels, cent, cnt, uniq, least_mut):
    small_idx = np.where(cnt < least_mut)[0]
    if small_idx.size == 0:
        return False, labels          # nothing to merge

    cid = small_idx[0]                # dissolve this cluster
    mask_src = labels == cid

    # centroids of all *other* clusters
    other_ids = np.delete(np.arange(len(uniq)), cid)
    other_cent = cent[other_ids]

    # distance matrix:  (n_src, k_other)
    dmat = np.linalg.norm(phi[mask_src][:, None, :] - other_cent[None, :, :],
                          axis=2)
    nearest = dmat.argmin(axis=1)
    labels[mask_src] = other_ids[nearest]      # per‑mutation reassignment
    return True, labels


# ──────────────────────────────────────────────────────────────────────
#  B. Dissolve ONE cluster from the closest centroid pair (< least_diff)
# ──────────────────────────────────────────────────────────────────────
def _merge_one_close(phi, labels, cent, cnt, uniq, least_diff):
    if len(cent) <= 1:
        return False, labels

    dist = squareform(pdist(cent, 'euclidean'))
    np.fill_diagonal(dist, np.inf)
    min_val = dist.min()
    if min_val >= least_diff:
        return False, labels

    i, j = np.unravel_index(dist.argmin(), dist.shape)
    # dissolve the *smaller* of the pair (ties → j)
    cid = i if cnt[i] < cnt[j] else j
    mask_src = labels == cid

    other_ids = np.delete(np.arange(len(uniq)), cid)
    other_cent = cent[other_ids]

    dmat = np.linalg.norm(phi[mask_src][:, None, :] - other_cent[None, :, :],
                          axis=2)
    nearest = dmat.argmin(axis=1)
    labels[mask_src] = other_ids[nearest]
    return True, labels

def _postprocess_clusters(phi, labels, least_mut, least_diff):
    """
    While‑loop that applies A then B, exactly ONE cluster
    dissolved per iteration.
    """
    merges = 0
    while merges < MAX_POST_MERGES:
        labels, cent, cnt, uniq = _reindex(phi, labels)

        # A. grow undersized cluster
        changed, labels = _merge_one_small(phi, labels, cent, cnt, uniq,
                                           least_mut)
        if changed:
            merges += 1
            continue            # re‑compute centroids before next step

        # B. merge closest centroids
        changed, labels = _merge_one_close(phi, labels, cent, cnt, uniq,
                                           least_diff)
        if changed:
            merges += 1
            continue

        break                   # no more merges possible

    labels, cent, _, _ = _reindex(phi, labels)
    return cent[labels].astype(np.float32), labels.astype(np.int64)

# ──────────────────────────────────────────────────────────────────────
# Penalised Ward routine + post‑processing
# ──────────────────────────────────────────────────────────────────────
def ward_penalised(phi: np.ndarray,
                   lambda_pen: float,
                   weights: Optional[np.ndarray] = None):
    """
    1. Build Ward dendrogram; choose K_opt(λ)
    2. Apply small & close‑cluster merging.
    """
    n, m = phi.shape
    if n < 2:
        return phi.copy(), np.zeros(n, np.int64)

    if weights is not None:
        if weights.shape != (n,):
            raise ValueError("weights must be (n,) vector")
        if np.any(weights < 0):
            raise ValueError("weights must be non‑negative")

    X = phi if weights is None else phi * np.sqrt(weights)[:, None]
    Z = linkage(X, method='ward', metric='euclidean')

    delta = Z[:, 2].astype(np.float64)
    cum_sse = np.concatenate([[0.0], np.cumsum(delta)])
    K_vals = np.arange(n, 0, -1, dtype=np.int32)
    obj = cum_sse + lambda_pen * (K_vals - 1)
    K_opt = min(int(K_vals[obj.argmin()]), 20)

    labels = fcluster(Z, t=K_opt, criterion='maxclust') - 1

    # -------- post‑processing thresholds ----------------------------
    least_mut = int(LEAST_MUT_FRAC*n) if LEAST_MUT_FRAC < 1 else int(LEAST_MUT_FRAC)
    least_mut = max(1, least_mut)
    least_diff = LEAST_DIFF_RAW * sqrt(m)

    phi_clean, labels = _postprocess_clusters(phi, labels,
                                              least_mut, least_diff)
    return phi_clean, labels

# ----------------------------------------------------------------------
# Log‑likelihood (unchanged numerically, but small tidy‑ups)
# ----------------------------------------------------------------------
def loglik_of_phi(phi_clean: np.ndarray,
                  df_phi_index: pd.MultiIndex,
                  pairs_df: pd.DataFrame,
                  tensors: Dict[str, np.ndarray],
                  eps: float = 1e-12) -> float:

    full_index = pairs_df.index
    locs = full_index.get_indexer(df_phi_index)
    keep = locs >= 0
    if keep.sum() == 0:
        return -np.inf

    phi_clean = phi_clean[keep]
    locs      = locs[keep]

    r, n = tensors['r'][locs], tensors['n'][locs]
    minor, total = tensors['minor'][locs], tensors['total'][locs]
    purity = tensors['purity'][locs] if tensors['purity'].ndim == 2 else tensors['purity']

    theta = (phi_clean * minor) / (2 * (1 - purity) + purity * total)
    theta = np.clip(theta, eps, 1 - eps)
    return float((r * np.log(theta) + (n - r) * np.log1p(-theta)).sum())


# ----------------------------------------------------------------------
# Stats helpers
# ----------------------------------------------------------------------
def safe_entropy(labels: np.ndarray) -> float:
    counts = np.bincount(labels)
    if counts.size == 1:                       # single cluster => 0
        return 0.0
    return float(entropy(counts, base=2))

def founder_distance(phi_clean: np.ndarray, labels: np.ndarray, rho: np.ndarray) -> float:
    cent = np.vstack([phi_clean[labels == k].mean(0)
                      for k in np.unique(labels)])
    # "founder" assumed at CCF = 1 across samples
    return float(np.min(np.linalg.norm(cent - rho, axis=1)))

def simulation_metrics(phi_clean, labels_pred, true_labels, params, rho):
    if np.isnan(phi_clean).any():
        return dict(ari=0, rdnc=1, rdcf=1, rmse=float('inf'))

    K = int(params['K'])
    ne = len(np.unique(labels_pred))
    rdnc = abs(ne - K) / K if K else abs(ne - K)

    ccf_e = phi_clean / rho
    uniq  = np.unique(labels_pred)
    cent  = np.vstack([ccf_e[labels_pred == u].mean(0) for u in uniq])
    target = uniq[np.linalg.norm(cent - 1.0, axis=1).argmin()]
    ce = (labels_pred == target).mean()

    csizes = np.array(params.get('cluster_size', []), dtype=float)
    ct = csizes[0] / csizes.sum() if csizes.size else 0
    rdcf = abs(ce - ct) / ct if ct else abs(ce - ct)

    ccf_true = np.array(params['cluster_ccf'])[true_labels]
    rmse = np.sqrt(((ccf_e - ccf_true) ** 2).mean())

    ari = adjusted_rand_score(true_labels, labels_pred)
    return dict(ari=ari, rdnc=rdnc, rdcf=rdcf, rmse=rmse)


# ----------------------------------------------------------------------
# Single‑lambda evaluation
# ----------------------------------------------------------------------
def evaluate_single_lambda(tsv_file: str, samples: List[str],
                           pairs_df: pd.DataFrame,
                           tensors: Dict[str, np.ndarray],
                           w1: float, w2: float,
                           truth_dict: Dict):

    try:
        df_phi = (pd.read_csv(tsv_file, sep='\t')
                  .pivot_table(index=['chromosome_index', 'position'],
                               columns='region', values='phi')
                  .reindex(columns=samples)
                  .dropna(how='all'))
        if df_phi.empty:
            return None

        phi = df_phi.to_numpy(np.float32)
        N   = phi.shape[0]
        M   = phi.shape[1]
        if N == 0:
            return None

        lam_pen = _parse_lambda(os.path.basename(tsv_file))

        # ------- weights ----------
        weights = None
        if USE_DEPTH_WEIGHT:
            pair_idx = pairs_df.index  # MultiIndex (chrom, pos)
            locs = pair_idx.get_indexer(df_phi.index)
            if (locs < 0).any():
                raise ValueError("φ rows not all present in pairs_df")
            weights = tensors['n'][locs]

        # ------- clustering --------
        phi_c, labels = ward_penalised(phi, lam_pen, weights)

        # ------- metrics -----------
        perf  = simulation_metrics(phi_c, labels,
                                   truth_dict['true_labels'],
                                   truth_dict['params'],
                                   truth_dict['rho'])

        loglik = loglik_of_phi(phi_c, df_phi.index, pairs_df, tensors)
        k_act  = len(np.unique(labels))
        bic    = -2.0*loglik if N <= 1 else -2.0*loglik + k_act*log(N)
        entr   = safe_entropy(labels)
        fdst   = founder_distance(phi_c, labels, tensors["purity"])
        becs   = 15 * bic / N + 100 * entr / np.log2(20) + 300 * fdst / (M ** 0.5)
        

        out = dict(source_lambda_file=os.path.basename(tsv_file),
                   **perf,
                   bec_score=becs,
                   log_likelihood=loglik,
                   bic_score=bic,
                   entropy=entr,
                   founder_distance=fdst,
                   num_clusters=k_act,
                   num_mutations=N)

        if EXPORT_BIG_ARRAYS:
            out.update(phi_clean=phi_c.tolist(),
                       labels=labels.tolist())
        return out

    except Exception as e:
        print(f"[WARNING] Failed {os.path.basename(tsv_file)}: {e}",
              file=sys.stderr)
        return None


# ----------------------------------------------------------------------
# Driver over simulation directory
# ----------------------------------------------------------------------
def process_all_lambdas_for_simulation(sim_dir: str, truth_dir: str,
                                       input_parent: str,
                                       w1: float, w2: float) -> List[Dict]:

    sim_name = os.path.basename(sim_dir)
    params_p = os.path.join(truth_dir, f"{sim_name}_params.json")
    clust_p  = os.path.join(truth_dir, f"{sim_name}_clusters.csv")
    if not (os.path.isfile(params_p) and os.path.isfile(clust_p)):
        raise FileNotFoundError(f"Truth files for {sim_name} not found.")

    params   = json.load(open(params_p))
    true_lbl = pd.read_csv(clust_p)["cluster_id"].to_numpy()
    truth    = dict(true_labels=true_lbl, params=params, rho=params['rho'])

    # tensors & samples
    from clipp2.preprocess import combine_patient_samples, add_to_df, build_tensor_arrays
    df_s = add_to_df(combine_patient_samples(os.path.join(input_parent, sim_name)))
    dic  = build_tensor_arrays(df_s)
    pairs_df = dic["pairs"].set_index(['chromosome_index', 'position'])
    tensors  = {k: dic[k] for k in ("r", "n", "minor", "total")}
    tensors["purity"] = dic["pur_arr"]
    samples  = dic["samples"]

    lambda_files = sorted(glob.glob(os.path.join(sim_dir, "lambda_*.tsv")))
    print(f"[INFO] {sim_name}: {len(lambda_files)} lambda files detected.")

    return [res for tsv in lambda_files
            if (res := evaluate_single_lambda(tsv, samples, pairs_df,
                                              tensors, w1, w2, truth))]


# ----------------------------------------------------------------------
# Batch runner
# ----------------------------------------------------------------------
def main(segment_idx: int, dirs = None):

    root_out  = '/data/Dropbox/GitHub/Simulation/output1'
    truth_dir = '/data/Dropbox/GitHub/Simulation/simulations_tree'
    input_dir = '/data/Dropbox/GitHub/Simulation/input'
    save_dir  = '/data/Dropbox/GitHub/Simulation/temp2/all_results_df'
    os.makedirs(save_dir, exist_ok=True)

    w1 = w2 = 10000
    per_segment = 809
    if dirs is not None:
        seg_dirs = json.load(open(dirs))
    else:
        sim_dirs = sorted(d for d in glob.glob(os.path.join(root_out, "*"))
                      if os.path.isdir(d))
        seg_dirs = sim_dirs[segment_idx*per_segment : (segment_idx+1)*per_segment]

    print(f"[INFO] Segment {segment_idx}: {len(seg_dirs)} dirs; saving to {save_dir}")
    print("-"*48)

    cols = ['source_lambda_file', 'ari', 'rdnc', 'rdcf', 'rmse',
            'bec_score', 'log_likelihood', 'bic_score', 'entropy',
            'founder_distance', 'num_clusters', 'num_mutations']

    for sim_dir in seg_dirs:
        name = os.path.basename(sim_dir)
        out_csv = os.path.join(save_dir, f"{name}_results.csv")
        if os.path.exists(out_csv) and dirs is None:
            continue  # skip if already processed
        print(f"▶ {sim_dir}")
        try:
            results = process_all_lambdas_for_simulation(sim_dir, truth_dir,
                                                         input_dir, w1, w2)
            if not results:
                print("   ⚠  no λ file succeeded")
                continue
            pd.DataFrame(results, columns=cols).to_csv(out_csv, index=False)
            print(f"   ✔  {len(results)} rows → {out_csv}")
        except Exception as e:
            print(f"   ✘  error: {e}", file=sys.stderr)

    print("-"*48, "\n[INFO] All done.")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--num', required=True, type=int,
                   help="segment index (0‑based)")
    p.add_argument('--dirs', default=None)
    args = p.parse_args()
    main(args.num, args.dirs)
