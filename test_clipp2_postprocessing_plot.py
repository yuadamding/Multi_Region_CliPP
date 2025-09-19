#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
φ̂ scatter (sample X vs sample Y), colored by final cluster assignment,
with point coordinates ALWAYS taken from the winning lambda_*.tsv
resolved via files in --save_dir produced by the selector.

Required files in --save_dir (default pathing):
  - selected_lambda.txt
  - <winning lambda_*.tsv>  (copied into save_dir by the selector)
  - selected_labels.csv     (fallback labels if --labels_csv not provided)
  - selected_prevalence_matrix.csv

You may override clusters with --labels_csv and choose the cluster column with --cluster_col.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

_FLOAT_RE = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')

def _high_contrast_palette() -> list[tuple]:
    base = [
        "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#7F7F7F", "#E31A1C",
        "#1F78B4", "#33A02C", "#FB9A99", "#A6CEE3", "#B2DF8A",
        "#CAB2D6", "#6A3D9A", "#FF7F00", "#B15928", "#8DD3C7"
    ]
    return [mpl.colors.to_rgb(c) for c in base]

def _colorize_labels(labels: np.ndarray) -> tuple[np.ndarray, dict]:
    uniq = np.unique(labels)
    palette = _high_contrast_palette()
    color_map = {int(u): palette[i % len(palette)] for i, u in enumerate(uniq)}
    colors = np.array([color_map[int(l)] for l in labels], dtype=float)
    return colors, color_map

def _parse_lambda_display(lambda_txt: Path) -> str | None:
    if not lambda_txt.exists():
        return None
    s = lambda_txt.read_text().strip()
    m = _FLOAT_RE.search(s)
    if m:
        try:
            return f"{float(m.group(0)):.4f}"
        except Exception:
            return s or None
    return s or None

def _parse_lambda_filename(lambda_txt: Path) -> str | None:
    if not lambda_txt.exists():
        return None
    s = lambda_txt.read_text().strip()
    m = re.search(r'(lambda_[0-9eE.+-]+\.tsv)', s)
    if m:
        return m.group(1)
    if s.endswith(".tsv"):
        return Path(s).name
    return None

def _resolve_winning_tsv(save_dir: Path) -> Path:
    lam_txt = save_dir / "selected_lambda.txt"
    fname = _parse_lambda_filename(lam_txt)
    if fname is None:
        raise FileNotFoundError(f"Could not parse lambda filename from {lam_txt}")
    local = save_dir / fname
    if local.exists():
        return local
    raw = lam_txt.read_text().strip()
    if raw.endswith(".tsv") and Path(raw).exists():
        return Path(raw)
    raise FileNotFoundError(f"Winning TSV not found in {save_dir}: expected {fname}")

def _read_centroids(save_dir: Path) -> pd.DataFrame:
    prev = save_dir / "selected_prevalence_matrix.csv"
    df = pd.read_csv(prev)
    if "cluster_id" not in df.columns:
        raise ValueError("selected_prevalence_matrix.csv must include 'cluster_id'.")
    return df.sort_values("cluster_id").reset_index(drop=True)

def _read_labels_default(save_dir: Path, cluster_col: str) -> pd.DataFrame:
    lab_path = save_dir / "selected_labels.csv"
    df = pd.read_csv(lab_path)
    req = {"chromosome_index", "position", cluster_col}
    if not req.issubset(df.columns):
        raise ValueError(f"{lab_path} must have columns: chromosome_index, position, {cluster_col}.")
    df = df[["chromosome_index", "position", cluster_col]].rename(columns={cluster_col: "cluster"})
    df = _canon_key(df)
    # drop potential duplicate rows on keys (keep last)
    df = df.drop_duplicates(subset=["chromosome_index", "position"], keep="last")
    return df

def _read_labels_override(labels_csv: Path, cluster_col: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    req = {"chromosome_index", "position", cluster_col}
    if not req.issubset(df.columns):
        raise ValueError(f"{labels_csv} must have columns: chromosome_index, position, {cluster_col}.")
    df = df[["chromosome_index", "position", cluster_col]].rename(columns={cluster_col: "cluster"})
    df = _canon_key(df)
    df = df.drop_duplicates(subset=["chromosome_index", "position"], keep="last")
    return df

def _read_labels(save_dir: Path, labels_csv: str | None, cluster_col: str) -> pd.DataFrame:
    if labels_csv:
        return _read_labels_override(Path(labels_csv), cluster_col)
    return _read_labels_default(save_dir, cluster_col)

def _read_raw_xy(raw_tsv: Path, xsample: str, ysample: str) -> pd.DataFrame:
    df = pd.read_csv(raw_tsv, sep="\t")
    required = {"chromosome_index", "position", "region", "phi_hat"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"{raw_tsv} missing columns: {missing}")
    pv = (
        df.pivot_table(index=["chromosome_index", "position"], columns="region", values="phi_hat")
          .sort_index()
    )
    for col in (xsample, ysample):
        if col not in pv.columns:
            raise ValueError(
                f"Sample '{col}' not found in {raw_tsv.name}. Available: {list(pv.columns)}"
            )
    out = pv[[xsample, ysample]].copy()
    out.columns = [xsample, ysample]
    return out  # index = (chromosome_index, position)

def _canon_key(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with (chromosome_index, position) coerced to canonical strings."""
    out = df.copy()
    out["chromosome_index"] = out["chromosome_index"].astype(str).str.strip()
    out["position"] = out["position"].astype(str).str.strip()
    return out

def _align_xy_with_labels(df_xy: pd.DataFrame, labels_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_df = df_xy.index.to_frame(index=False)
    idx_df = _canon_key(idx_df)
    labels_df = _canon_key(labels_df)

    merged = idx_df.merge(labels_df, on=["chromosome_index", "position"], how="left")
    if merged["cluster"].isna().any():
        missing = int(merged["cluster"].isna().sum())
        total = merged.shape[0]
        if missing == total:
            raise ValueError("No coordinate rows matched labels by (chromosome_index, position).")
        keep_mask = ~merged["cluster"].isna()
        merged = merged[keep_mask].reset_index(drop=True)
        df_xy = df_xy.iloc[keep_mask.to_numpy(bool)]

    x = df_xy.iloc[:, 0].to_numpy(float)
    y = df_xy.iloc[:, 1].to_numpy(float)
    lbl = merged["cluster"].astype(int).to_numpy()
    return x, y, lbl

def _make_legend_for_clusters(ax, labels: np.ndarray, color_map: dict[int, tuple], title: str = "Cluster"):
    uniq = np.unique(labels)
    if uniq.size == 0 or uniq.size > 30:
        return None
    handles, texts = [], []
    for u in sorted(map(int, uniq)):
        col = color_map[int(u)]
        h = ax.scatter([], [], s=28, c=[col])
        handles.append(h)
        texts.append(str(u))
    leg = ax.legend(handles, texts, title=title, loc="upper left", frameon=True)
    leg.get_frame().set_alpha(0.9)
    return leg

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", required=True, help="Folder with selection outputs")
    ap.add_argument("--xsample", required=True)
    ap.add_argument("--ysample", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--vline", type=float, default=1.0)
    ap.add_argument("--hline", type=float, default=1.0)
    # NEW: override clusters if you want
    ap.add_argument("--labels_csv", default=None, help="Optional CSV with custom clusters for each mutation")
    ap.add_argument("--cluster_col", default="cluster", help="Column name in labels CSV to use as cluster id (default: 'cluster')")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    if not save_dir.exists():
        raise FileNotFoundError(f"--save_dir not found: {save_dir}")

    # Resolve inputs purely from save_dir
    lam_txt  = save_dir / "selected_lambda.txt"
    raw_tsv  = _resolve_winning_tsv(save_dir)
    df_prev  = _read_centroids(save_dir)
    labels   = _read_labels(save_dir, args.labels_csv, args.cluster_col)
    df_rawxy = _read_raw_xy(raw_tsv, args.xsample, args.ysample)
    x, y, lbl = _align_xy_with_labels(df_rawxy, labels)

    # Centroids for X markers (still from prevalence matrix)
    for col in (args.xsample, args.ysample):
        if col not in df_prev.columns:
            raise ValueError(f"Sample '{col}' not found in selected_prevalence_matrix.csv.")
    cent_xy = df_prev[[args.xsample, args.ysample]].to_numpy(float)

    # Plot
    fig = plt.figure(figsize=(6.0, 6.6), dpi=150)
    ax  = fig.add_subplot(111)

    pt_colors, color_map = _colorize_labels(lbl)
    ax.scatter(x, y, s=10, c=pt_colors, alpha=0.8, rasterized=True)

    ax.scatter(cent_xy[:, 0], cent_xy[:, 1], marker='x', s=90, linewidths=2, c='red')
    for i, (cx, cy) in enumerate(cent_xy):
        ax.text(cx, cy, f" {i}", color='red', fontsize=9, weight='bold', ha='left', va='center')

    ax.axvline(args.vline, linestyle='--', linewidth=1, color='black')
    ax.axhline(args.hline, linestyle='--', linewidth=1, color='black')
    ax.grid(True, which='both', linestyle=':', linewidth=0.6, alpha=0.5)

    _make_legend_for_clusters(ax, lbl, color_map)

    lam_str = _parse_lambda_display(lam_txt)
    title = f"φ̂ Comparison: {args.xsample} vs {args.ysample}"
    if lam_str:
        title += f"\nλ = {lam_str}"
    ax.set_title(title)
    ax.set_xlabel(f"φ̂ ({args.xsample})")
    ax.set_ylabel(f"φ̂ ({args.ysample})")

    # Force axes to start at -0.1; add padding to top/right
    x_max = np.nanmax(x) if x.size else 0.0
    y_max = np.nanmax(y) if y.size else 0.0
    pad_x = 0.05 * (x_max if x_max > 0 else 1.0)
    pad_y = 0.05 * (y_max if y_max > 0 else 1.0)
    ax.set_xlim(-0.1, x_max + pad_x)
    ax.set_ylim(-0.1, y_max + pad_y)
    ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    outpath = Path(args.out) if args.out else (save_dir / f"phi_scatter_{args.xsample}_vs_{args.ysample}.png")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[OK] Saved: {outpath}")
    print(f"[INFO] Coordinates drawn from: {raw_tsv}")
    src_lab = args.labels_csv if args.labels_csv else (save_dir / "selected_labels.csv")
    print(f"[INFO] Labels source: {src_lab}")
    print(f"[INFO] Inputs: {save_dir/'selected_prevalence_matrix.csv'}, {save_dir/'selected_lambda.txt'}")

if __name__ == "__main__":
    main()
