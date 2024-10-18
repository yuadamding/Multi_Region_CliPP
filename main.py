import pandas as pd
from snv2 import *
import ray


if __name__ == "__main__":
    ray.shutdown()
    ray.init()
    _output_root = 'multi_clipp_simulation_data_March4/'
    df = pd.read_csv(_output_root + 'simulation_data_cluster_5_region_6_read_depth_100_replica_2.tsv', sep = '\t')

    n = 100
    m = 6
    df = df.iloc[0: (n * m), :]


    rho = 1
    gamma = [0.25 * i for i in range(20)]
    omega = 1
    max_iteration = 10

    res = [ADMM2.remote(df, rho, gamma[i], omega, n, m, max_iteration)[3] for i in range(20)]
    res = ray.get(res)
    ray.shutdown()
    print(res)