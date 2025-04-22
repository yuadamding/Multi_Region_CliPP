import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_clipp2_ccf(sample, processed_data_dir='processed_data'):
    """
    Load clustering outputs for ACC<sample> and plot a normalized CCF scatter.

    Parameters
    ----------
    sample : int or str
        The numeric portion of the sample code (e.g. 21) or full code 'ACC21'.
    processed_data_dir : str
        Root directory where 'processed_data/ACC<sample>/' lives.
    """
    # Construct sample code
    if isinstance(sample, int) or sample.isdigit():
        sample_code = f'ACC{int(sample)}'
    else:
        sample_code = sample

    # Build paths
    out_dir = os.path.join(processed_data_dir, sample_code)
    paths = {
        'labels'  : os.path.join(out_dir, f'{sample_code}label.pkl'),
        'ccf'     : os.path.join(out_dir, f'{sample_code}ccf.pkl'),
        'centroid': os.path.join(out_dir, f'{sample_code}centroid.pkl'),
        'purity'  : os.path.join(out_dir, f'{sample_code}purity.pkl'),
    }

    # Load data
    with open(paths['labels'],   'rb') as f: lab         = pickle.load(f)
    with open(paths['ccf'],      'rb') as f: ccf_corrected = pickle.load(f)
    with open(paths['centroid'], 'rb') as f: centroid    = pickle.load(f)
    with open(paths['purity'],   'rb') as f: purity      = pickle.load(f)

    # Normalize by purity
    x = ccf_corrected[:, 1] / purity[1]
    y = ccf_corrected[:, 0] / purity[0]

    # Plot
    plt.figure(figsize=(6,6))
    sc = plt.scatter(
        x, y,
        c=lab,
        cmap='viridis',
        s=2,
        alpha=0.5
    )

    plt.xlim(-0.02, 3)
    plt.ylim(-0.02, 3)
    plt.xlabel(f'{sample_code} pre CCF',  fontsize=14, fontweight='bold')
    plt.ylabel(f'{sample_code} post CCF', fontsize=14, fontweight='bold')
    plt.title('CliPP2',             fontsize=18, fontweight='bold')

    # Legend for clusters
    handles, labels = sc.legend_elements(prop='colors', alpha=0.5)
    plt.legend(
        handles=handles,
        labels=labels,
        title='Clusters',
        loc='best'
    )

    plt.grid(False)
    plt.show()




def plot_clipp2_cp(sample, processed_data_dir='processed_data'):
    """
    Load CLiPP2 outputs for ACC<sample> and plot the scatter
    with vertical/horizontal purity lines labeled as 'pre_purity' and 'post_purity'.
    
    Parameters
    ----------
    sample : int or str
        The numeric part of the sample code (e.g. 9 or '9').
    processed_data_dir : str
        Root directory where `processed_data/ACC<sample>/` lives.
    """
    # Build sample code and paths
    sample = str(sample)
    sample_code = f'ACC{sample}'
    out_dir = os.path.join(processed_data_dir, sample_code)
    
    # Load files
    with open(os.path.join(out_dir, f'{sample_code}label.pkl'),   'rb') as f:
        lab = pickle.load(f)
    with open(os.path.join(out_dir, f'{sample_code}ccf.pkl'),     'rb') as f:
        ccf_corrected = pickle.load(f)
    with open(os.path.join(out_dir, f'{sample_code}centroid.pkl'),'rb') as f:
        centroid = pickle.load(f)
    with open(os.path.join(out_dir, f'{sample_code}purity.pkl'),  'rb') as f:
        purity = pickle.load(f)
    
    # Plot
    plt.figure(figsize=(6,6))
    sc = plt.scatter(
        ccf_corrected[:, 1],
        ccf_corrected[:, 0],
        c=lab,         
        cmap='viridis',
        s=2,
        alpha=0.5
    )
    
    ax = plt.gca()
    # draw guide‐lines
    ax.axvline(purity[1], color='red',   linestyle='--', linewidth=1)
    ax.axhline(purity[0], color='blue',  linestyle='--', linewidth=1)
    
    # annotate them horizontally
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    
    ax.text(
        purity[1], ylim[0],
        f"pre_purity={purity[1]:.2f}",
        color='red',
        va='bottom', ha='center',
        backgroundcolor='white'
    )
    ax.text(
        xlim[0], purity[0],
        f"post_purity={purity[0]:.2f}",
        color='blue',
        va='center', ha='left',
        backgroundcolor='white'
    )
    
    # labels, title, legend
    plt.xlabel(f'{sample_code} pre CP',  fontsize=14, fontweight='bold')
    plt.ylabel(f'{sample_code} post CP', fontsize=14, fontweight='bold')
    plt.title('CliPP2',              fontsize=18, fontweight='bold')
    
    handles, labels = sc.legend_elements(prop="colors", alpha=0.5)
    plt.legend(handles, labels, title="Clusters", loc="best")
    
    plt.xlim(-0.02, 1.4)
    plt.ylim(-0.02, 1.4)
    plt.grid(False)
    plt.show()
