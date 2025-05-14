import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_phi_hat_comparison(df, region_x, region_y, point_size=2):
    """
    Plot φ̂ comparison between two regions, colored by cluster label,
    and mark each cluster’s centroid with its mean φ (from df['phi']).

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ['chromosome_index', 'position', 'region',
        'phi_hat', 'label', 'phi'].
    region_x : str
        Region name to plot on the x-axis.
    region_y : str
        Region name to plot on the y-axis.
    point_size : float, optional
        Scatter point size (default: 2).
    """
    df = pd.read_csv('res.tsv', sep='\t')

    # pivot φ̂ into wide form
    phi_wide = df.pivot_table(
        index=['chromosome_index','position'],
        columns='region',
        values='phi_hat'
    )
    if region_x not in phi_wide.columns or region_y not in phi_wide.columns:
        raise ValueError(f"Regions must be among {list(phi_wide.columns)}")

    x = phi_wide[region_x]
    y = phi_wide[region_y]

    # get one label + one phi per mutation
    meta = (
        df
        .drop_duplicates(subset=['chromosome_index','position'])
        .set_index(['chromosome_index','position'])
        [['label','phi']]
    )
    meta = meta.reindex(phi_wide.index)

    # only keep those with both φ̂s and a known label
    mask = x.notna() & y.notna() & meta['label'].notna()
    x = x[mask]
    y = y[mask]
    lab = meta.loc[mask, 'label'].astype(int)
    phi_true = meta.loc[mask, 'phi']

    # scatter
    plt.figure(figsize=(6,6))
    sc = plt.scatter(
        x, y,
        c=lab,
        cmap='viridis',
        s=point_size,
        alpha=0.5
    )
    ax = plt.gca()
    ax.set_xlabel(f'cp({region_x})', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'cp({region_y})', fontsize=14, fontweight='bold')
    ax.set_title(f'cp: {region_x} vs {region_y}', fontsize=18, fontweight='bold')

    # legend by cluster
    handles, labels = sc.legend_elements(prop="colors", alpha=0.5)
    ax.legend(handles, labels, title="Label", loc="best")

    # compute & mark centroids
    for lbl in np.unique(lab):
        sel = (lab == lbl)
        cx, cy = x[sel].mean(), y[sel].mean()
        phi_mean = phi_true[sel].mean()
        # big red cross
        ax.scatter(cx, cy, marker='x', s=80, c='red', linewidths=2)
        # annotate with φ̄ value
        ax.text(
            cx, cy,
            f"cluster {lbl}",
            fontsize=12,
            fontweight='bold',
            va='bottom',
            ha='right'
        )

    ax.grid(False)
    plt.tight_layout()
    plt.show()
