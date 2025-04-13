import numpy as np
import pandas as pd
from clipp2.clipp2 import * 
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

class Mutation:
    def __init__(self, cluster, tumor_purity, cp):
        self.cluster = cluster
        elements = [4, 3, 2, 1]
        probabilities = [0.05, 0.15, 0.2, 0.6]
        self.major_cn = np.random.choice(elements, p=probabilities)
        self.minor_cn = 1
        self.total_reads = np.random.poisson(100, 1)[0]
        self.tumor_purity = tumor_purity
        self.cp = cp
        
        self.b = self.major_cn
        self.c = self.get_c()
        self.reads = self.get_read()
    
    def get_c(self):
        return min(self.b / ((1 - self.tumor_purity) * 2 + self.tumor_purity * (self.major_cn + self.minor_cn)), 1)
    
    def get_read(self):
        c = self.c
        cp = self.cp
        total_reads = self.total_reads
        # reads = total_reads * cp * c + np.random.normal(0, 0.01 * total_reads)
        reads = np.random.binomial(total_reads, cp * c)
        return reads

def generate_data(n, m, cp, cluster):
    regions = ['R' + str(i % m + 1) for i in range(n * m)]

    ref_counts = []
    alt_counts = []
    normal_cn = [2] * n * m
    major_cn = []
    minor_cn = []
    if m == 4:
        tumour_purity = [0.75, 0.65, 0.85, 0.7]
    elif m == 2:
        tumour_purity = [0.75, 0.65]
    clusters = [cluster] * n * m

    for i in range(n * m):
        mut = Mutation(cluster, tumour_purity[i % m], cp[i % m])
        ref_counts.append(mut.total_reads - mut.reads)
        alt_counts.append(mut.reads)
        major_cn.append(mut.major_cn)
        minor_cn.append(mut.minor_cn)

    data = {
        'mutation': np.repeat(np.arange((cluster - 1) * n, cluster * n), m),
        'region': regions,
        'ref_counts': ref_counts,
        'alt_counts': alt_counts,
        'normal_cn': normal_cn,
        'major_cn': major_cn,
        'minor_cn': minor_cn,
        'tumour_purity': tumour_purity * n,
        'cluster': clusters
    }
    
    df = pd.DataFrame(data)
    return df

def ccf_plot_between_sample(filename, n, m):
    df = pd.read_csv(filename, sep = '\t')
    c_mat = get_c_mat(df)
    purity_mat = get_purity_mat(df)
    read_mat = get_read_mat(df)
    total_read_mat = get_total_read_mat(df)
    ccf = read_mat / (total_read_mat * purity_mat * c_mat)
    cluster = [df['cluster'][m * i] for i in range(n)]
    
    sample_pairs = list(itertools.combinations(range(m), 2))
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    
    for i, (sample1, sample2) in enumerate(sample_pairs[:6]):  # Limit to the first 6 pairs for the 3x2 grid
        ax = axes.flat[i]
        sns.scatterplot(x=ccf[:, sample1], y=ccf[:, sample2], hue=cluster, palette='viridis', alpha=0.7, ax=ax)
        ax.set_title(f'Sample {sample1} vs Sample {sample2}')
        ax.set_xlabel(f'Sample {sample1} CCF')
        ax.set_ylabel(f'Sample {sample2} CCF')
        ax.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()


# # Example usage
# df1 = generate_data(n=25, m=4, cp=[0.99, 0.99, 0.99, 0.99], cluster=1)
# df2 = generate_data(n=25, m=4, cp=[0.99, 0.01, 0.01, 0.99], cluster=2)
# df3 = generate_data(n=25, m=4, cp=[0.99, 0.01, 0.99, 0.01], cluster=3)
# df4 = generate_data(n=25, m=4, cp=[0.1, 0.99, 0.01, 0.99], cluster=4)

# combined_df = pd.concat([df1, df2, df3, df4], axis=0)
# combined_df.to_csv('test4.tsv', sep='\t', index=False)
# ccf_plot_between_sample('test4.tsv', 100, 4)