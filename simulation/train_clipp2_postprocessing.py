from __future__ import annotations
import glob, json, os, re, sys
from math import log, sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.metrics import adjusted_rand_score
from concurrent.futures import ProcessPoolExecutor, as_completed

# conda activate ml1
# python train_clipp2_postprocessing_2.py --num 0 --workers 20
# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
USE_DEPTH_WEIGHT   = False
SCALE_LAMBDA       = 10.0
EXPORT_BIG_ARRAYS  = False

LEAST_MUT_FRAC     = 0.05    # fraction of N; <1 ⇒ relative, ≥1 ⇒ absolute
LEAST_DIFF_RAW     = 0.02    # raw per‑sample gap, scaled by √m later
MAX_POST_MERGES    = 100     # safety to avoid infinite loop

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
    for i, _ in enumerate(uniq):
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

# ─────────────────────────────────────────────────────────────────────
# Cluster‑prevalence helpers
# ─────────────────────────────────────────────────────────────────────
def _cluster_prevalence(phi_clean: np.ndarray,
                        labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
      cp_mat … shape (K, M) – mean CP per (cluster, sample)
      uniq   … length‑K     – cluster IDs corresponding to rows
    """
    uniq = np.unique(labels)
    K, M = len(uniq), phi_clean.shape[1]
    cp   = np.zeros((K, M), dtype=phi_clean.dtype)
    for i, k in enumerate(uniq):
        cp[i] = phi_clean[labels == k].mean(0)
    return cp, uniq

def _merge_two_clusters(labels: np.ndarray,
                        cid_keep: int,
                        cid_merge: int) -> np.ndarray:
    """
    Relabel cid_merge → cid_keep and make the label set contiguous.
    """
    labels = labels.copy()
    labels[labels == cid_merge] = cid_keep
    _, inv = np.unique(labels, return_inverse=True)
    return inv

# ─────────────────────────────────────────────────────────────────────
# Post‑BECS fixes for Super‑clones and Small‑clones
# ─────────────────────────────────────────────────────────────────────
def handle_superclusters(phi_clean: np.ndarray,
                         labels: np.ndarray,
                         purity: np.ndarray,
                         max_clonal_frac: float = 0.40) -> np.ndarray:
    """
    Multi‑sample super‑cluster consolidation.
    """
    while True:
        cp_mat, uniq = _cluster_prevalence(phi_clean, labels)
        cp_mat = cp_mat / purity  # normalise by purity
        K, M = cp_mat.shape

        # (i) super‑clone present?
        if not (cp_mat > 1.0).any():
            break

        # (ii) too few clusters?
        if K <= 2:
            break

        # (iii) clonal cluster & its fraction
        d2_one = np.linalg.norm(cp_mat - 1.0, axis=1)      # L2 to 𝟙
        clonal_idx = d2_one.argmin()
        clonal_frac = np.bincount(labels)[clonal_idx] / labels.size
        if clonal_frac > max_clonal_frac:
            break

        # merge two clusters with largest overall CP
        cp_mean = np.linalg.norm(cp_mat, axis=1)
        top2 = np.argsort(cp_mean)[-2:]
        cid_keep  = int(uniq[top2[1]])
        cid_merge = int(uniq[top2[0]])
        labels = _merge_two_clusters(labels, cid_keep, cid_merge)

    return labels

def handle_small_clones(phi_clean: np.ndarray,
                        labels: np.ndarray,
                        max_clonal_frac: float = 0.15) -> np.ndarray:
    """
    Multi‑sample small‑clone consolidation.
    """
    while True:
        cp_mat, uniq = _cluster_prevalence(phi_clean, labels)
        K = cp_mat.shape[0]
        if K <= 2:
            break

        d2_one = np.linalg.norm(cp_mat - 1.0, axis=1)      # L2 to 𝟙
        clonal_idx = d2_one.argmin()
        clonal_frac = np.bincount(labels)[clonal_idx] / labels.size
        if clonal_frac > max_clonal_frac:
            break

        cp_mean = np.linalg.norm(cp_mat, axis=1)
        top2 = np.argsort(cp_mean)[-2:]
        cid_keep  = int(uniq[top2[1]])
        cid_merge = int(uniq[top2[0]])
        labels = _merge_two_clusters(labels, cid_keep, cid_merge)

    return labels

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
# NEW: save final labels for a single lambda file
# ----------------------------------------------------------------------
def _save_final_labels(tsv_file: str,
                       df_phi_index: pd.MultiIndex,
                       labels_final: np.ndarray,
                       assign_out_dir: str) -> str:
    """
    Writes per-mutation final cluster IDs to CSV.
    Path: <assign_out_dir>/<sim_name>/<lambda_base>_clusters.csv
    """
    sim_name = os.path.basename(os.path.dirname(tsv_file))
    base     = os.path.splitext(os.path.basename(tsv_file))[0]  # 'lambda_xxx'
    out_dir  = os.path.join(assign_out_dir, sim_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base}_clusters.csv")

    idx_df = df_phi_index.to_frame(index=False)
    idx_df['cluster_id'] = labels_final.astype(np.int64)
    idx_df.to_csv(out_path, index=False)
    return out_path

# ----------------------------------------------------------------------
# Single‑lambda evaluation (now also writes final cluster assignments)
# ----------------------------------------------------------------------
def evaluate_single_lambda(tsv_file: str, samples: List[str],
                           pairs_df: pd.DataFrame,
                           tensors: Dict[str, np.ndarray],
                           w1: float, w2: float,
                           truth_dict: Dict,
                           assign_out_dir: Optional[str] = None):

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

        # ------- final label consolidation (post-BECS) ----------
        labels = handle_superclusters(phi_c, labels, tensors["purity"])
        labels = handle_small_clones(phi_c, labels)
        labels_final, _, _, _ = _reindex(phi_c, labels)  # ensure 0-based contiguous

        # ------- persist per-test assignments ----------
        if assign_out_dir is not None:
            _ = _save_final_labels(tsv_file, df_phi.index, labels_final, assign_out_dir)

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
                       labels=labels_final.tolist())  # final labels

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
                                       w1: float, w2: float,
                                       assign_out_dir: Optional[str] = None) -> List[Dict]:

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

    results: List[Dict] = []
    for tsv in lambda_files:
        res = evaluate_single_lambda(tsv, samples, pairs_df, tensors, w1, w2, truth,
                                     assign_out_dir=assign_out_dir)
        if res is not None:
            results.append(res)
    return results

# ----------------------------------------------------------------------
# NEW: single-sim-dir worker, suitable for ProcessPoolExecutor
# ----------------------------------------------------------------------
def process_one_sim_dir(sim_dir: str,
                        truth_dir: str,
                        input_dir: str,
                        w1: float, w2: float,
                        save_dir: str,
                        assign_out_dir: str,
                        cols: List[str],
                        force: bool = False,
                        dirs_is_none: bool = True) -> Tuple[str, int, str, Optional[str]]:
    """
    Returns: (sim_name, nrows_written, status, error_message_if_any)
    status in {'ok','skipped','empty','error'}
    """
    sim_name = os.path.basename(sim_dir)
    out_csv  = os.path.join(save_dir, f"{sim_name}_results.csv")

    # Original logic: skip existing only when dirs is None; allow override via --force
    if os.path.exists(out_csv) and dirs_is_none and not force:
        return sim_name, 0, "skipped", None

    try:
        results = process_all_lambdas_for_simulation(sim_dir, truth_dir, input_dir, w1, w2,
                                                     assign_out_dir=assign_out_dir)
        if not results:
            return sim_name, 0, "empty", None

        os.makedirs(save_dir, exist_ok=True)
        pd.DataFrame(results, columns=cols).to_csv(out_csv, index=False)
        return sim_name, len(results), "ok", None
    except Exception as e:
        return sim_name, 0, "error", str(e)

# ----------------------------------------------------------------------
# Batch runner (parallel)
# ----------------------------------------------------------------------
def main(segment_idx: int, dirs = None, workers: int = 1, force: bool = False,
         assign_out_override: Optional[str] = None):

    # Keep BLAS libraries from over-subscribing threads inside each worker
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    root_out  = '/data/GitHub/Simulation/output1'
    truth_dir = '/data/GitHub/Simulation/simulations_tree'
    input_dir = '/data/GitHub/Simulation/input1'
    save_dir  = '/data/GitHub/Simulation/temp2/all_results_df'
    os.makedirs(save_dir, exist_ok=True)

    # where to put per-test cluster assignments
    assign_out_dir = (assign_out_override if assign_out_override
                      else os.path.join(save_dir, "cluster_assignments"))
    os.makedirs(assign_out_dir, exist_ok=True)

    w1 = w2 = 10000

    seg_dirs = sorted(d for d in glob.glob(os.path.join(root_out, "*"))
                          if os.path.isdir(d))

    print(f"[INFO] Segment {segment_idx}: {len(seg_dirs)} dirs; saving to {save_dir}")
    print("-"*56)

    cols = ['source_lambda_file', 'ari', 'rdnc', 'rdcf', 'rmse',
            'bec_score', 'log_likelihood', 'bic_score', 'entropy',
            'founder_distance', 'num_clusters', 'num_mutations']

    # parallel execution over simulation directories
    n_ok = n_skip = n_empty = n_err = 0
    futures = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as ex:
        for sim_dir in seg_dirs:
            fut = ex.submit(process_one_sim_dir, sim_dir, truth_dir, input_dir,
                            w1, w2, save_dir, assign_out_dir, cols, force, dirs_is_none=(dirs is None))
            futures.append(fut)

        for fut in as_completed(futures):
            sim_name, nrows, status, err = fut.result()
            if status == "ok":
                n_ok += 1
                print(f"✔ {sim_name:>32}  ({nrows} rows)")
            elif status == "skipped":
                n_skip += 1
                print(f"↷ {sim_name:>32}  (skipped existing)")
            elif status == "empty":
                n_empty += 1
                print(f"⚠ {sim_name:>32}  (no λ file succeeded)")
            else:
                n_err += 1
                print(f"✘ {sim_name:>32}  (error: {err})", file=sys.stderr)

    print("-"*56)
    print(f"[INFO] Done. ok={n_ok}, skipped={n_skip}, empty={n_empty}, error={n_err}")

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--num', required=True, type=int,
                   help="segment index (0‑based)")
    p.add_argument('--dirs', default=None,
                   help="JSON file with explicit list of sim_dirs (disables skip-on-exist unless --force).")
    p.add_argument('--workers', type=int, default=os.cpu_count() or 1,
                   help="number of parallel worker processes")
    p.add_argument('--force', action='store_true',
                   help="recompute even if per-sim results CSV exists")
    p.add_argument('--assign_out', default=None,
                   help="override directory to save per-test cluster assignment CSVs")
    args = p.parse_args()
    main(args.num, args.dirs, args.workers, args.force, args.assign_out)
