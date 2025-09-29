from clipp2.core import *
from clipp2.preprocess import *
import json

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def calculate_ccf_from_df(df: pd.DataFrame, min_one: bool = True, cap01: bool = False) -> pd.DataFrame:
    """Computes CCF from a DataFrame containing all necessary information."""
    required_cols = ['alt_count', 'ref_count', 'total_cn', 'purity']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain columns: {', '.join(required_cols)}")

    out_df = df.copy()
    depth = out_df['alt_count'] + out_df['ref_count']
    vaf = np.divide(out_df['alt_count'], depth, out=np.full_like(depth, np.nan, dtype=float), where=depth > 0)
    purity = out_df['purity']
    total_cn = out_df['total_cn']
    m_star = (vaf / purity) * (purity * total_cn + 2 * (1 - purity))
    m_round = np.round(m_star)
    if min_one:
        m_round = np.maximum(1, m_round)
    out_df['multiplicity'] = m_round.astype('Int64')
    
    multiplicity = out_df['multiplicity'].astype(float)
    multiplicity[multiplicity <= 0] = np.nan
    ccf_numerator = vaf * (purity * total_cn + 2 * (1 - purity))
    ccf_denominator = multiplicity * purity
    ccf = np.divide(ccf_numerator, ccf_denominator, out=np.full_like(ccf_numerator, np.nan), where=ccf_denominator!=0)
    if cap01:
        ccf = np.clip(ccf, 0, 1)
    out_df['CCF'] = ccf
    return out_df


def plot_ccf_matrix(df_with_ccf: pd.DataFrame):
    """
    Generates a lower-triangular matrix of scatter plots with a smaller point size.
    """
    # --- 1. Reshape the data for plotting ---
    sns.set_style("white")
    df_with_ccf['variant_id'] = df_with_ccf['chromosome_index'].astype(str) + ':' + df_with_ccf['position'].astype(str)
    
    plot_df = df_with_ccf.pivot_table(
        index='variant_id', columns='sample', values='CCF'
    )
    cluster_map = df_with_ccf[['variant_id', 'cluster_id']].drop_duplicates().set_index('variant_id')
    plot_df = plot_df.merge(cluster_map, left_index=True, right_index=True).dropna()

    if plot_df.empty:
        print("No variants with CCF values across all samples to plot.")
        return
        
    samples = sorted([col for col in plot_df.columns if col != 'cluster_id'])
    n = len(samples)
    if n < 2:
        print("Cannot create a matrix plot with fewer than 2 samples.")
        return

    # --- 2. Dynamically determine axis limits ---
    data_min = plot_df[samples].min().min()
    data_max = plot_df[samples].max().max()
    view_min, view_max = min(0, data_min), max(1, data_max)
    padding = (view_max - view_min) * 0.05
    lower_bound, upper_bound = view_min - padding, view_max + padding

    # --- 3. Create the subplot matrix ---
    fig, axes = plt.subplots(n, n, figsize=(2 * n, 2 * n), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    # --- 4. Populate the subplots ---
    clusters = sorted(plot_df['cluster_id'].unique())
    colors = sns.color_palette("deep", n_colors=len(clusters))
    color_map = dict(zip(clusters, colors))
    point_colors = plot_df['cluster_id'].map(color_map)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j < i: # Strict lower triangle
                sample_y, sample_x = samples[i], samples[j]
                
                # ***MODIFIED LINE***
                # The 's' parameter is reduced from 25 to 5 for smaller points.
                ax.scatter(plot_df[sample_x], plot_df[sample_y], c=point_colors, s=5, alpha=0.8)
                
                ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], 'r--', alpha=0.5, zorder=0)
                
                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1.5)

                ax.set_xlim(lower_bound, upper_bound)
                ax.set_ylim(lower_bound, upper_bound)
                
                if j == 0 and i > 0:
                    ax.set_ylabel(sample_y, fontsize=10)
                if i == n - 1:
                    ax.set_xlabel(sample_x, fontsize=10)

                ax.tick_params(axis='x', labelsize=8, labelrotation=90)
                ax.tick_params(axis='y', labelsize=8)
            else:
                ax.set_visible(False)

    # --- 5. Create a shared legend ---
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                              label=f'Cluster {c}', markerfacecolor=color, markersize=8)
                       for c, color in color_map.items()]
    fig.legend(handles=legend_elements, title='Cluster ID', loc='upper right')
    fig.suptitle('Pairwise CCF Comparison Matrix', fontsize=16, y=0.95)
    plt.show()