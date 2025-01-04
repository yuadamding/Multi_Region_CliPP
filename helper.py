from snv import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import imageio.v2 as imageio
import subprocess
import glob
import shutil

def convert_to_plyclonevi(df):
    df.rename(columns={
    "mutation": "mutation_id",
    "region": "sample_id",
    "tumour_purity": "tumour_content"
    }, inplace=True)
    return df

def convert_back_to_original(df):
    df.rename(columns={
        "mutation_id": "mutation",
        "sample_id": "region",
        "tumour_content": "tumour_purity"
    }, inplace=True)
    return df

def get_num_sample(df):
    return int(df['region'].nunique())

def get_num_snvs(df):
    return int(len(df) / get_num_sample(df))

def ccf_plot(df):
    ccf = df['ccf']
    m = get_num_sample(df)
    n = get_num_snvs(df)
    cluster = df['cluster']
    sample1 = [i * m  for i in range(n)]
    sample2 = [i * m + 1  for i in range(n)]
    scatter = sns.scatterplot(
        x=ccf[sample1],
        y=ccf[sample2],
        hue=cluster,
        palette='viridis',
        alpha=1
    )

    scatter.set_title(f'Sample 1 vs Sample 2')
    scatter.set_xlabel(f'Sample 1 CCF')
    scatter.set_ylabel(f'Sample 2 CCF')
    plt.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_inverse(y):
    if np.any((y <= 0) | (y >= 1)):
        raise ValueError("Input to sigmoid_inverse must be in the range (0, 1) exclusive.")
    
    return np.log(y / (1 - y))
    
def convert_p_to_cp(arr, n, m):
    arr = np.reshape(arr, [n, m])
    cp = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            cp[i, j] = sigmoid(arr[i, j])
    return cp

def check_p_v(p, v, n, m):
    sets = {i for i in range(n)}
    combinations_2 = list(itertools.combinations(sets, 2))
    pairs_mapping = {combination: index for index, combination in enumerate(combinations_2)}
    pairs_mapping_inverse = {index: combination for index, combination in enumerate(combinations_2)}
    for i in range(len(combinations_2)):
        index = pairs_mapping[combinations_2[i]]
        start_v = index * m 
        end_v = (index + 1) * m
        l1, l2 = pairs_mapping_inverse[index]
        start_p1 = l1 * m
        start_p2 = l2 * m
        end_p1 = l1 * m + m
        end_p2 = l2 * m + m
        if np.all(p[start_p1 : end_p1] - p[start_p2 : end_p2] == v[start_v : end_v]):
            continue
        elif np.all(p[start_p1 : end_p1] - p[start_p2 : end_p2] == -v[start_v : end_v]):
            continue
        else:
            raise("Error detected. p1: {p[start_p1 : end_p1]}, p2: {p[start_p2 : end_p2]}, v:{v[start_v : end_v]}")
    
    return 0


def create_gif(image_sequence, output_file, duration=2):
    """
    Generate a GIF from a sequence of images.

    Parameters:
        image_sequence (list of str): Paths to the image files in sequence.
        output_file (str): Path for the output GIF file.
        duration (float): Duration of each frame in seconds (default is 0.5s).
    """
    with imageio.get_writer(output_file, mode='I', duration=duration) as writer:
        for image_path in image_sequence:
            image = imageio.imread(image_path)
            writer.append_data(image)
    print(f"GIF saved as {output_file}")
    
def change_df_to_pyclonevi(df):
    df.rename(columns={
        'mutation': 'mutation_id',
        'region': 'sample_id',
        'tumour_purity': 'tumour_content'
    }, inplace=True)
    df.to_csv('pyclonevi_input.tsv', sep='\t', index=False)

def get_pyclonevi_output(df):
    change_df_to_pyclonevi(df)
    subprocess.run(['pyclone-vi', 'fit', 'pyclonevi_input.csv', ' -o', 'pyclonevi_intemediate_output.h5', '-c 40 -d beta-binomial -r 10'])
    subprocess.run(['pyclone-vi write-results-file -i', 'pyclonevi_intemediate_output.h5', '-o', 'pyclonevi_output.tsv'])
    df = pd.read_csv('pyclonevi_output.tsv')
    
def get_file_for_clipp(df):

    dfs_by_region = {region: data for region, data in df.groupby('region')}
    n = len(dfs_by_region)
    # Create an empty matrix of zeros
    matrix = np.zeros((n, n), dtype=int)
    for key in dfs_by_region.keys():
        df = dfs_by_region[key]

        # Create df1 with mutation, ref_counts, and alt_counts
        df1 = df[['mutation', 'ref_counts', 'alt_counts']].copy()
        df1['mutation'] = [1 for i in range(1, len(df1) + 1)]
        df1.rename(columns={
            'ref_counts': 'ref_count',
            'alt_counts': 'alt_count'
        }, inplace=True)
        df1.rename(columns={'mutation': 'chromosome_index'}, inplace=True)
        df1['position'] = [5 * i + 2 for i in range(1, len(df1) + 1)]
        df1 = df1[['chromosome_index', 'position', 'alt_count', 'ref_count']]
        df1.to_csv(key + '_snv.txt', sep='\t', index=False)

        # Create df2 with mutation, major_cn, minor_cn, and sum_cn
        df2 = df[['mutation', 'major_cn', 'minor_cn']].copy()
        df2['mutation'] =[1 for i in range(1, len(df2) + 1)]
        df2['total_cn'] = df2['major_cn'] + df2['minor_cn']
        df2.rename(columns={'mutation': 'chromosome_index'}, inplace=True)
        df2['start_position'] = [5 * i + 1 for i in range(1, len(df2) + 1)]
        df2['end_position'] = [5 * i + 4 for i in range(1, len(df2) + 1)]
        df2 = df2[['chromosome_index', 'start_position', 'end_position', 'major_cn', 'minor_cn', 'total_cn']]

        df2.to_csv(key + '_cna.txt', sep='\t', index=False)

        # Get the first element of the 'tumour_purity' column
        purity = df['tumour_purity'].iloc[0]

        # Open a file in write mode and save the number
        with open(key + '_purity.txt', 'w') as file:
            file.write(str(purity))
        subprocess.run(['docker', 'run', '-v', '$(pwd):/Sample', 'clipp', 'python3', '/CliPP/run_clipp_main.py', '-i', '/Sample/test', '/Sample/sample_snv.txt', '/Sample/sample_cna.txt', '/Sample/sample_purity.txt'])
        
        file = glob.glob('test/final_result/Best_lambda/*mutation_assignments*')
        df = pd.read_csv(file[0], sep='\t')
        cluster_index = df['cluster_index'].values
        
        # Fill the matrix
        for i in range(n):
            for j in range(n):
                if cluster_index[i] == cluster_index[j]:
                    matrix[i, j] += 1
        shutil.rmtree('test')

    return matrix / len(dfs_by_region.keys())


