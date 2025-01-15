'''
----------------------------------------------------------------------
----------------------------------------------------------------------
This file contains functions for multi-region CliPP
Authors: Yu Ding
Date: 10/29/2024
Email: yding1995@gmail.com; yding4@mdanderson.org
----------------------------------------------------------------------
-----------------------------------------------------------------------
'''
import numpy as np
import itertools
import torch
import ray
from scipy.optimize import fsolve
import pandas as pd

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def adjust_array(arr, epsilon=1e-10):
    arr = np.array(arr)  # Ensure input is a numpy array
    arr[arr <= 0] = epsilon
    arr[arr >= 1] = 1 - epsilon
    return arr

def inverse_sigmoid(y):
    y = adjust_array(y)
    return np.log(y / (1 - y))

def sigmoid_derivative(x):
  return sigmoid(x) * (1 - sigmoid(x))

def sigmoid_second_derivative(x):
  return sigmoid_derivative(x) * (1 - 2 * sigmoid(x))

def a_mat_generator(v1, v2, n, m):
    # This function generates A_l matrix, please see notes for definition
    # @param v1 & v2, two distinct SNVs
    # @param n: number of SNVs
    # @param m: number of samples
    
    if n < 2 or m < 1:
        raise("The number of SNVs or number of samples are wrong.")
    
    temp1 = np.zeros(n)
    temp2 = np.zeros(n)
    temp1[v1] = 1
    temp2[v2] = 1
    a_mat = np.kron(temp1 - temp2, np.diag(np.ones(m)))
    return a_mat

def a_trans_a_mat_generator_quick(n, m):
    # This function is a fast version to generates \sum_{l\in\Omega}A^T_l %*% A_l matrix, please see notes for definition
    # @param n: number of SNVs
    # @param m: number of samples
    init_line = [0 for i in range(n * m)]
    init_line[0] = n - 1
    for i in range(1, n * m):
        if i % m == 0:
            init_line[i] = -1
    res = np.zeros([n * m, n * m])
    for i in range(n * m):
        res[i, :] = init_line[(n * m - i):(n * m)] + init_line[0:(n * m - i)]
    return res


def matmul_by_torch(amat, bmat):
    amat = torch.from_numpy(amat)
    bmat = torch.from_numpy(bmat)
    res_mat = torch.matmul(amat, bmat)
    return res_mat.numpy()

def sum_by_torch(mat):
    mat = torch.from_numpy(mat)
    res_mat = torch.sum(mat)
    return res_mat.numpy()

def get_major_cn(df):
    n = len(np.unique(df.mutation))
    m = sum(df.mutation == np.unique(df.mutation)[0])
    res = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            index = i * m + j
            res[i, j] = df.iloc[index , :].major_cn
    return res

def get_b_mat(df):
    n = len(np.unique(df.mutation))
    m = sum(df.mutation == np.unique(df.mutation)[0])
    res = np.zeros([n, m])
    tumor_cn_mat = get_tumor_cn_mat(df)
    normal_cn_mat = get_normal_cn_mat(df)
    purity_mat = get_purity_mat(df)
    major_cn_mat = get_major_cn(df)
    read_mat = get_read_mat(df)
    total_read_mat = get_total_read_mat(df)
    
    # for i in range(n):
    #     for j in range(m):
    #         index = i * m + j
    #         res[i, j] = df.iloc[index , :].multiplicity
    
    for i in range(n):
        for j in range(m):
            temp = read_mat[i, j] * (purity_mat[i, j] * tumor_cn_mat[i, j] + normal_cn_mat[i, j] * (1 - purity_mat[i, j]))\
                / (total_read_mat[i, j] * purity_mat[i, j])
            temp = 1 if np.round(temp) == 0 else np.round(temp)
            res[i, j] = np.min([temp, major_cn_mat[i, j]])
    return res

def get_tumor_cn_mat(df):
    n = len(np.unique(df.mutation))
    m = sum(df.mutation == np.unique(df.mutation)[0])
    res = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            index = i * m + j
            res[i, j] = df.iloc[index , ].major_cn + df.iloc[index , ].minor_cn
    return res

def get_normal_cn_mat(df):
    n = len(np.unique(df.mutation))
    m = sum(df.mutation == np.unique(df.mutation)[0])
    res = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            index = i * m + j
            res[i, j] = df.iloc[index , ].normal_cn
    return res

def get_purity_mat(df):
    n = len(np.unique(df.mutation))
    m = sum(df.mutation == np.unique(df.mutation)[0])
    res = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            index = i * m + j
            res[i, j] = df.iloc[index , ].tumour_purity
    return res

def get_read_mat(df):
    n = len(np.unique(df.mutation))
    m = sum(df.mutation == np.unique(df.mutation)[0])
    res = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            index = i * m + j
            res[i, j] = df.iloc[index , ].alt_counts
    return res

def get_total_read_mat(df):
    n = len(np.unique(df.mutation))
    m = sum(df.mutation == np.unique(df.mutation)[0])
    res = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            index = i * m + j
            res[i, j] = df.iloc[index , ].alt_counts + df.iloc[index , ].ref_counts
    return res

def get_c_mat(df):

    tumor_cn_mat = get_tumor_cn_mat(df)
    normal_cn_mat = get_normal_cn_mat(df)
    purity_mat = get_purity_mat(df)
    b_mat = get_b_mat(df)
    c_mat = b_mat / ((1 - purity_mat) * normal_cn_mat + purity_mat * tumor_cn_mat)
    return c_mat

def get_loglikelihood(p, c_mat, read_mat, total_read_mat):
    cp = sigmoid(p)
    prop = cp * c_mat
    if np.any(prop < 0) or np.any(prop > 1):
        print(prop)
    return sum_by_torch(read_mat * np.log(prop) + (total_read_mat - read_mat) * np.log(1 - prop))

def get_v_mat(v, y, rho, combinations, pairs_mapping, n, m):
    
    v_tilde = v + y / rho
    res = np.zeros([n * m])
    for i in range(len(combinations)):
        pair = combinations[i]
        index_v = pairs_mapping[pair]
        start_v = index_v * m 
        end_v = (index_v + 1) * m
        a_mat = a_mat_generator(pair[0], pair[1], n, m)
        res = res + matmul_by_torch(a_mat.T, v_tilde[start_v: end_v])
    return res

def get_objective_function(p, v, y, rho,combinations, pairs_mapping, c_mat, read_mat, total_read_mat, n, m, gamma, omega):
    res = -get_loglikelihood(np.reshape(p, [n, m]), c_mat, read_mat, total_read_mat)
    for i in range(len(combinations)):
        pair = combinations[i]
        index_v = pairs_mapping[pair]
        start_v = index_v * m 
        end_v = (index_v + 1) * m
        a_mat = a_mat_generator(pair[0], pair[1], n, m)
        temp = v[start_v: end_v] - matmul_by_torch(a_mat, p)
        res = res + 0.5 * rho * matmul_by_torch(temp.T, temp)
        res = res + matmul_by_torch(y[start_v: end_v].T, temp)
        temp = v[start_v: end_v]
        res = res + gamma * omega[i] * np.sqrt(matmul_by_torch(temp.T, temp))
    return res

def get_grad_of_objective_function(v, y, rho,combinations, pairs_mapping, c_mat, read_mat, total_read_mat, n, m):
    v_mat = rho * get_v_mat(v, y, rho, combinations, pairs_mapping, n, m)
    a_mat = rho * a_trans_a_mat_generator_quick(n, m)

    def grad_of_objective_function(p):
        p = np.reshape(p, [n, m])
        cp_prime = sigmoid_derivative(p)
        cp = sigmoid(p)
        prop = cp * c_mat
        if np.any(prop < 0) or np.any(prop > 1) or np.nan in prop:
            print(prop)
        grad = cp_prime * (read_mat - total_read_mat * prop) / (cp * (1 - prop))
        grad = np.reshape(grad, [n*m])
        grad = -grad + matmul_by_torch(a_mat, np.reshape(p, [n*m])) - v_mat
        return grad

    return grad_of_objective_function

def get_objective_function_p(v, y, rho,combinations, pairs_mapping, c_mat, read_mat, total_read_mat, n, m):
    v_tilde = v + y / rho
    
    def objective_function_p(p):
        p = np.reshape(p, [n, m])
        res = -get_loglikelihood(p, c_mat, read_mat, total_read_mat)
        for i in range(len(combinations)):
            pair = combinations[i]
            index_v = pairs_mapping[pair]
            start_v = index_v * m 
            end_v = (index_v + 1) * m
            a_mat = a_mat_generator(pair[0], pair[1], n, m)
            temp = v_tilde[start_v: end_v] - matmul_by_torch(a_mat, np.reshape(p, [n*m]))
            res = res + 0.5 * rho * matmul_by_torch(temp.T, temp)
        return res
    
    return objective_function_p
    
def update_v(index_v, pairs_mapping_inverse, p_vec, y, n, m, rho, omega, gamma):
    start_y = index_v * m
    end_y = (index_v + 1) * m
    l1, l2 = pairs_mapping_inverse[index_v]
    a_mat = a_mat_generator(l1, l2, n, m)
    temp = matmul_by_torch(a_mat, p_vec) - y[start_y: end_y] / rho
    norm = np.sqrt(matmul_by_torch(temp.T, temp))
    if norm > gamma * omega / rho:
        v = (1 - gamma * omega / (rho * norm)) * temp
    else:
        v = np.zeros(m)
    
    return v
    
def update_y(y, v, index_v, pairs_mapping_inverse, p_vec, n, m, rho):
    
    l1, l2 = pairs_mapping_inverse[index_v]
    a_mat = a_mat_generator(l1, l2, n, m)
    # start_p_l1 = l1 * m
    # end_p_l1 = (l1 + 1) * m
    # start_p_l2 = l2 * m
    # end_p_l2 = (l2 + 1) * m
    return y + rho * (v - matmul_by_torch(a_mat, p_vec))

def dis_cluster(p, v, n, m, combinations, pairs_mapping, gamma):
    dic = {i : i for i in range(n)}
    res = np.zeros([n, n])
    for i in range(len(combinations)):
        # l1, l2 = combinations[i]
        # start_p_l1 = l1 * m
        # end_p_l1 = (l1 + 1) * m
        # start_p_l2 = l2 * m
        # end_p_l2 = (l2 + 1) * m
        # cp1 = sigmoid(p[start_p_l1: end_p_l1])
        # cp2 = sigmoid(p[start_p_l2: end_p_l2])
        # v_index = cp1 - cp2
        index_v = pairs_mapping[combinations[i]]
        start_v = index_v * m 
        end_v = (index_v + 1) * m
        v_index = v[start_v : end_v]
        temp = np.sqrt(matmul_by_torch(v_index.T, v_index) / m)
        if temp < 1e-2:
            res[combinations[i]] = 1
        
    for i in range(n - 1):
        for j in range(i + 1, n):
                if res[i, j] == 1:
                    dic[j] = dic[i]
              
    print(f"Gamma: {gamma}, clusters : {dic.values()}")      
            
    return dic

@ray.remote(num_returns=1)
def ADMM(df, rho, gamma, omega, n, m, max_iteration):
    """
    Alternating Direction Method of Multipliers (ADMM) for optimization.
    
    Parameters:
    df: pd.DataFrame
        Input dataframe containing mutation data.
    rho: float
        Rho value.
    gamma: float
        Gamma value.
    omega: float
        Omega value.
    n: int
        Number of SNVs.
    m: int
        Number of samples.
    max_iteration: int
        Maximum number of iterations.
    
    Returns:
    dict
        Dictionary containing the clusters.
    """
    # Initialize combinations and mappings
    sets = {i for i in range(n)}
    combinations_2 = list(itertools.combinations(sets, 2))
    pairs_mapping = {combination: index for index, combination in enumerate(combinations_2)}
    pairs_mapping_inverse = {index: combination for index, combination in enumerate(combinations_2)}

    # Get matrices
    read_mat = get_read_mat(df)
    total_read_mat = get_total_read_mat(df)
    c_mat = get_c_mat(df)

    # Initialize variables
    #12/12/2024
    # p = inverse_sigmoid(read_mat / (total_read_mat * c_mat))
    p = np.zeros([n * m])
    p = p.reshape([n * m])
    v = np.zeros([len(combinations_2) * m])
    y = np.ones([len(combinations_2) * m])
    omega = np.ones([len(combinations_2)])
    k = 0
    # obj_values = []
    Flag = True
    
    # ADMM
    while k < max_iteration and Flag:
        k += 1

        # Calculate current objective function
        # curr_objective_function = get_objective_function(
        #     p, v, y, rho, combinations_2, pairs_mapping, c_mat, read_mat, total_read_mat, n, m, gamma, omega
        # )
        # print(f"Iteration {k}, Objective Function: {curr_objective_function}")
        # obj_values.append(curr_objective_function)
        
        fp = get_grad_of_objective_function(
            v, y, rho, combinations_2, pairs_mapping, c_mat, read_mat, total_read_mat, n, m
        )
        p = fsolve(fp, p) 
        
        for i in range(len(combinations_2)):
            pair = combinations_2[i]
            index_v = pairs_mapping[pair]
            start_v = index_v * m
            end_v = (index_v + 1) * m
            # temp = np.sqrt(matmul_by_torch(v[start_v: end_v].T, v[start_v: end_v]))
            # temp = 1e-10 if temp == 0 else temp
            # omega[i] = pow(temp, -0.2)
            v[start_v: end_v] = update_v(index_v, pairs_mapping_inverse, p, y, n, m, rho, omega[i], gamma)                
            y[start_v: end_v] = update_y(y[start_v: end_v], v[start_v: end_v], i, pairs_mapping_inverse, p, n, m, rho)

    cls = dis_cluster(p, v, n, m, combinations_2, pairs_mapping, gamma)
    loglik = get_loglikelihood(np.reshape(p, [n, m]), c_mat, read_mat, total_read_mat)
    bic = -2 * loglik
    dof = len(np.unique(list(cls.values()))) * m
    bic = bic + np.log(n) * len(np.unique(list(cls.values()))) * m
    
    # PostProcess
    cp = sigmoid(p)
    mutation_df = pd.DataFrame(
        {
            'mutation': [i for i in range(n)],
            'cluster': [cls[i] for i in range(n)],
            'cp_sqrt': [np.sqrt(matmul_by_torch(cp[i * m: (i + 1) * m].T, cp[i * m: (i + 1) * m])) for i in range(n)]
        }
    )
    cluster_cp = mutation_df.groupby('cluster')['cp_sqrt'].mean()
    num_cluster = len(cluster_cp)
    largest_mean_cluster = cluster_cp.idxmax()
    clonal_fraction = mutation_df['cluster'].value_counts(normalize=True)[largest_mean_cluster]
    # Filter 1: deal with superclusters
    # If the max CP > 1 & the sample has > 2 clusters & the current clonal fraction <= 0.4:
    # Merge the two clusters with the largest CP values
    while cluster_cp.max() > 1 and num_cluster > 2 and clonal_fraction <= 0.4:
        # Get the two clusters with the largest CP values
        largest_cluster = cluster_cp.idxmax()
        cluster_cp.drop(largest_cluster, inplace=True)
        second_largest_cluster = cluster_cp.idxmax()

        # Update the cluster assignment
        mutation_df['cluster'] = mutation_df['cluster'].replace(largest_cluster, second_largest_cluster)
        cluster_cp = mutation_df.groupby('cluster')['cp_sqrt'].mean()
        num_cluster = len(cluster_cp)
        largest_mean_cluster = cluster_cp.idxmax()
        clonal_fraction = mutation_df['cluster'].value_counts(normalize=True)[largest_mean_cluster]
    # Filter 2: deal with small clones, i.e., the clonal cluster has a small number of mutations
    # If the sample has > 2 clusters & the current clonal fraction <= 0.15:
    # Merge the two clusters with the largest CP values
    while num_cluster > 2 and clonal_fraction <= 0.15:
        # Get the two clusters with the largest CP values
        largest_cluster = cluster_cp.idxmax()
        cluster_cp.drop(largest_cluster, inplace=True)
        second_largest_cluster = cluster_cp.idxmax()

        # Update the cluster assignment
        mutation_df['cluster'] = mutation_df['cluster'].replace(largest_cluster, second_largest_cluster)
        cluster_cp = mutation_df.groupby('cluster')['cp_sqrt'].mean()
        num_cluster = len(cluster_cp)
        largest_mean_cluster = cluster_cp.idxmax()
        clonal_fraction = mutation_df['cluster'].value_counts(normalize=True)[largest_mean_cluster]
    # Filter 3: deal with adjacent clusters, i.e., cluster with similar CP values
    # If the sample has > 2 clusters & if CP values between any two clusters < 0.1
    # Merge those two clusters together
    if num_cluster > 2:
        cluster_pairs = list(itertools.combinations(cluster_cp.index, 2))
        pairwise_differences = {}
        for (cluster1, cluster2) in cluster_pairs:
            diff = abs(cluster_cp[cluster1] - cluster_cp[cluster2])
            pairwise_differences[(cluster1, cluster2)] = diff
        min_diff = min(pairwise_differences.values())
        while num_cluster > 2 and min_diff < 0.1:
            # Get the two clusters with the smallest difference in CP values
            keys_with_value = [key for key, value in pairwise_differences.items() if value == min_diff]
            cluster1 = keys_with_value[0][0]
            cluster2 = keys_with_value[0][1]

            # Update the cluster assignment
            mutation_df['cluster'] = mutation_df['cluster'].replace(cluster1, cluster2)
            cluster_cp = mutation_df.groupby('cluster')['cp_sqrt'].mean()
            num_cluster = len(cluster_cp)
            if num_cluster <= 2:
                break
            cluster_pairs = list(itertools.combinations(cluster_cp.index, 2))
            pairwise_differences = {}
            for (cluster1, cluster2) in cluster_pairs:
                diff = abs(cluster_cp[cluster1] - cluster_cp[cluster2])
                pairwise_differences[(cluster1, cluster2)] = diff
            min_diff = min(pairwise_differences.values())
            
            largest_mean_cluster = cluster_cp.idxmax()
            clonal_fraction = mutation_df['cluster'].value_counts(normalize=True)[largest_mean_cluster]
    # Filter 4: deal with small subclones, i.e., subclonal clusters with a small number of mutations
    # If a subclone has cluster_num < 0.05 * total mutaton
    # Merge this subclone with its closest cluster
    cluster_num = mutation_df['cluster'].value_counts()
    while cluster_num.min() < 0.05 * n:
        # Get the subclone with the smallest number of mutations
        min_cluster = cluster_num.idxmin()
        cluster_num = cluster_num.drop(min_cluster)
        second_min_cluster = cluster_num.idxmin()

        # Update the cluster assignment
        mutation_df['cluster'] = mutation_df['cluster'].replace(min_cluster, second_min_cluster)
        cluster_cp = mutation_df.groupby('cluster')['cp_sqrt'].mean()
        cluster_num = mutation_df['cluster'].value_counts()
        num_cluster = len(cluster_cp)
        largest_mean_cluster = cluster_cp.idxmax()
        clonal_fraction = mutation_df['cluster'].value_counts(normalize=True)[largest_mean_cluster]
        
    cls  = mutation_df['cluster'].to_list()
    mutation_df = pd.DataFrame(
        {
            'mutation': [i for i in range(n)],
            'cluster': cls,
        }
    )
    for j in range(m):
        mutation_df[f'p_{j}'] = [p[i * m + j] for i in range(n)]
        mutation_df[f'p_{j}'] = mutation_df.groupby('cluster')[f'p_{j}'].transform('mean')
    p = np.reshape(mutation_df.iloc[:, 2:].values, [n * m])
    
    return [p, v, y, bic, loglik, gamma, dof, cls]


def find_gamma(res, purity, n, m):
    A_score_lst = []
    for i in range(len(res)):
        temp1 = sigmoid(res[i][0])
        temp1 = np.reshape(temp1, (n, m))
        temp2 = res[i][7]
        df = pd.DataFrame(
            {
                'cluster': temp2
            }
        )
        for j in range(m):
            name = 'cp' + str(j)
            df[name] = temp1[:, j]
        
        max_cp = 0
        max_cp_ind = 0
        for j in range(n):
            cp = df.iloc[0, :][1: (m + 1)]
            cp_value = np.matmul(np.transpose(cp), cp) 
            if cp_value > max_cp:
                max_cp_ind = j
        max_cp = df.iloc[max_cp_ind, :][1: (m + 1)]
        A_score = np.matmul(np.transpose(max_cp - purity), max_cp - purity) / np.matmul(np.transpose(purity), purity)
        A_score_lst.append(np.sqrt(A_score))
        
    A_score_lst = np.array(A_score_lst)
    if any((A_score_lst) < 0.05):
        best_ind = np.max(np.where(A_score_lst < 0.05))
        return(res[best_ind])
    elif all((A_score_lst) > 0.01):
        best_ind = np.argmin(A_score_lst)
        return(res[best_ind])
    else:
        print("Cannot select lambda given current criterion.")
        
def find_gamma_single_region(res, purity, n, m = 1):
    A_score_lst = []
    for i in range(len(res)):
        temp1 = sigmoid(res[i][0])
        temp2 = res[i][7]
        df = pd.DataFrame(
            {
                'cluster': temp2,
                'cp' : temp1
            }
        )
        
        max_cp = max(df['cp'])
        A_score = abs(max_cp - purity) / purity
        A_score_lst.append(A_score)
        
    A_score_lst = np.array(A_score_lst)
    if any((A_score_lst) < 0.05):
        best_ind = np.max(np.where(A_score_lst < 0.05))
        return(res[best_ind])
    elif all((A_score_lst) > 0.01):
        best_ind = np.argmin(A_score_lst)
        return(res[best_ind])
    else:
        print("Cannot select lambda given current criterion.")




