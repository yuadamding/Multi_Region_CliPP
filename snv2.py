'''----------------------------------------------------------------------
This file contains functions for multi-region CliPP
Authors: Yu Ding
Date: 10/16/2024
Email: yding1995@gmail.com; yding4@mdanderson.org
----------------------------------------------------------------------
-----------------------------------------------------------------------
'''
import numpy as np
import scipy as sci
import itertools
import time
import ray

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
    
    for i in range(n):
        for j in range(m):
            temp = read_mat[i, j] * (purity_mat[i, j] * tumor_cn_mat[i, j] + normal_cn_mat[i, j] * (1 - purity_mat[i, j]))/ (total_read_mat[i, j] * purity_mat[i, j])
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

def get_prop_mat(df, cp = None, mapped_cp = None):
    if cp is None and mapped_cp is None:
        raise("Please input either cp or mapped_cp.")
    
    if cp is None:
        cp = sci.stats.norm.cdf(mapped_cp)
    
    b_mat = get_b_mat(df)
    tumor_cn_mat = get_tumor_cn_mat(df)
    normal_cn_mat = get_normal_cn_mat(df)
    purity_mat = get_purity_mat(df)
    res = cp * b_mat / ((1 - purity_mat) * normal_cn_mat + purity_mat * tumor_cn_mat)
    return res

def get_loglikelihood(p, b_mat, tumor_cn_mat, normal_cn_mat, purity_mat, read_mat, total_read_mat):
    cp = sci.stats.norm.cdf(p)
    prop = cp * b_mat / ((1 - purity_mat) * normal_cn_mat + purity_mat * tumor_cn_mat)
    return np.sum(read_mat * np.log(prop) + (total_read_mat - read_mat) * np.log(1 - prop))
    

def get_obj_one_snv_p(index, p_vec, v, y, rho, pairs_mapping, b_vec, tumor_cn_vec, normal_cn_vec, purity_vec, read_vec, total_read_vec, n, m):
    v_tilde = v + y / rho
    
    def obj_one_snv_p(p):
        cp = sci.stats.norm.cdf(p)
        prop = cp * b_vec / ((1 - purity_vec) * normal_cn_vec + purity_vec * tumor_cn_vec)
        res = -1 * np.sum(read_vec * np.log(prop) + (total_read_vec - read_vec) * np.log(1 - prop))
        for i in range(n):
            if i != index:
                pair = (index, i) if index < i else (i, index)
                index_v = pairs_mapping[pair]
                start_v = index_v * m 
                end_v = (index_v + 1) * m
                start_p = i * m
                end_p = (i + 1) * m
                temp = v_tilde[start_v : end_v] - p + p_vec[start_p: end_p]
                res = res + 0.5 * rho * np.matmul(temp.T, temp) / 2 
    
        return res
    
    return obj_one_snv_p

def update_v(index_v, pairs_mapping_inverse, p_vec, y, m, rho, omega, gamma):
    start_y = index_v * m
    end_y = (index_v + 1) * m
    l1, l2 = pairs_mapping_inverse[index_v]
    start_p_l1 = l1 * m
    end_p_l1 = (l1 + 1) * m
    start_p_l2 = l2 * m
    end_p_l2 = (l2 + 1) * m
    temp = p_vec[start_p_l1:end_p_l1] - p_vec[start_p_l2: end_p_l2] - y[start_y: end_y] / rho
    norm = np.sqrt(np.matmul(temp.T, temp))
    if norm >= gamma * omega / rho:
        v = (1 - gamma * omega / (rho * norm)) * temp
    else:
        v = np.zeros(m)
    
    return v
    # def obj_v(v):
    #     temp1 = v - temp
    #     res = 0.5 * np.matmul(temp1.T, temp1)
    #     res = res + gamma * omega * np.sqrt(np.matmul(v.T, v)) / rho
    #     return res
    
    # return obj_v
    
def update_y(y, v, index_v, pairs_mapping_inverse, p_vec, m, rho):
    l1, l2 = pairs_mapping_inverse[index_v]
    start_p_l1 = l1 * m
    end_p_l1 = (l1 + 1) * m
    start_p_l2 = l2 * m
    end_p_l2 = (l2 + 1) * m
    return y + rho * (v - p_vec[start_p_l1:end_p_l1] + p_vec[start_p_l2: end_p_l2])



def dis_cluster(v, n, m, combinations, pairs_mapping):
    dic = {i : i for i in range(n)}
    res = np.zeros([n, n])
    for i in range(len(combinations)):
        index_v = pairs_mapping[combinations[i]]
        start_v = index_v * m 
        end_v = (index_v + 1) * m
        v_index = v[start_v : end_v]
        if np.matmul(v_index.T, v_index) < 0.01:
            res[combinations[i]] = 1
        
    for i in range(n - 1):
        for j in range(i + 1, n):
                if res[i, j] == 1:
                    dic[j] = dic[i]
              
    print(dic)      
            
    return dic

@ray.remote(num_returns = 4)
def ADMM(df, rho, gamma, omega, n, m, max_iteration):
    
    sets = {i for i in range(n)}
    combinations_2 = list(itertools.combinations(sets, 2))
    dic1 = {}
    dic2 = {}
    index = 0
    for i in range(len(combinations_2)):
        combination = combinations_2[i]
        dic1[combination] = index
        dic2[index] = combination
        index = index + 1
    pairs_mapping = dic1
    pairs_mapping_inverse = dic2

    b_mat = get_b_mat(df)
    tumor_cn_mat = get_tumor_cn_mat(df)
    normal_cn_mat = get_normal_cn_mat(df)
    purity_mat = get_purity_mat(df)
    read_mat = get_read_mat(df)
    total_read_mat = get_total_read_mat(df)
    
    p = np.zeros([n*m])
    v = np.ones([len(combinations_2) * m])
    y = np.ones([len(combinations_2) * m])

    from scipy.optimize import minimize
    k = 0
    while(k < max_iteration):
        k = k + 1
        for i in range(n):
            start = i * m
            end = (i + 1) * m
            p0 = p[start: end]
            res = minimize(get_obj_one_snv_p(i, 
                                             p, 
                                             v, 
                                             y, 
                                             rho, 
                                             pairs_mapping, 
                                             b_mat[i, :], 
                                             tumor_cn_mat[i, :], 
                                             normal_cn_mat[i, :], 
                                             purity_mat[i, :], 
                                             read_mat[i, :], 
                                             total_read_mat[i, :], 
                                             n, 
                                             m), 
                       p0, 
                       method='nelder-mead',
                       options={'xatol': 1e-2, 'disp': False})
            p[start: end] = res.x

        for i in range(len(combinations_2)):
            pair = combinations_2[i]
            index_v = pairs_mapping[pair]
            start_v = index_v * m
            end_v = (index_v + 1) * m
            v0 = v[start_v: end_v]
            v[start_v: end_v] = update_v(index_v, pairs_mapping_inverse, p, y, m, rho, omega, gamma)
            y[start_v: end_v] = update_y(y[start_v: end_v], v[start_v: end_v], i, pairs_mapping_inverse, p, m, rho)
            
    cls = dis_cluster(v, n, m, combinations_2, pairs_mapping)
    bic = -2 * get_loglikelihood(np.reshape(p, [n, m]), b_mat, tumor_cn_mat,normal_cn_mat, purity_mat, read_mat, total_read_mat)
    bic = bic + np.log(n) * len(np.unique(cls.values)) * m
    
    return p, v, y, bic




