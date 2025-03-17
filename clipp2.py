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
from scipy.special import logit
from scipy.special import expit
import scipy as sp

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

def mat_inverse_by_torch(mat):
    mat = torch.from_numpy(mat)
    res_mat = torch.inverse(mat)
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

def theta(w, c):
    return (np.exp(w) * c) / (1 + np.exp(w))

def linear_evaluate(x, a, b):
    return a * x + b

def linear_approximate(c):
    w = [-4, -1.8, 1.8, 4]
    actual_theta = theta(w, c)
    
    b_1 = (actual_theta[1] - actual_theta[0]) / (w[1] - w[0])
    a_1 = actual_theta[0] - w[0] * b_1
    b_2 = (actual_theta[2] - actual_theta[1]) / (w[2] - w[1])
    a_2 = actual_theta[1] - w[1] * b_2
    b_3 = (actual_theta[3] - actual_theta[2]) / (w[3] - w[2])
    a_3 = actual_theta[2] - w[2] * b_3
    
    return {
        'w_cut': [-1.8, 1.8],
        'coef': [b_1, a_1, b_2, a_2, b_3, a_3],
    }

def get_linear_approximation(c_mat):
    n = c_mat.shape[0]
    m = c_mat.shape[1]
    wcut = []
    coef = []
    for i in range(n):
        for j in range(m):
            wcut.append(linear_approximate(c_mat[i, j])['w_cut'])
            coef.append(linear_approximate(c_mat[i, j])['coef'])
    return wcut, coef

def get_a_and_b_mat(p_old, total_read_mat, c_mat, read_mat, n, m, linearApprox):
    p = p_old.reshape([n, m])
    cp = sigmoid(p)
    prop = cp * c_mat
    a_mat = np.zeros([n, m])
    b_mat = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            # tag1, tag2, tag3, tag4 = 0, 0, 0, 0
            # temp = linearApprox[(i, j)]
            # w_cut = temp['w_cut']
            
            # if p[i, j] < w_cut[0]:
            #     tag1 = 1
            # else:
            #     tag3 = 1
            
            # if p[i, j] > w_cut[0]:
            #     tag2 = 1
            # else:
            #     tag4 = 1
                
            # coefs = temp['coef']
            
            if prop[i, j] >= 1:
                prop[i, j] = 1 - 1e-10
            
            coef2 = c_mat[i, j] / 10
            coef1 = c_mat[i, j] / 2 
            
            a_mat[i, j] = np.sqrt(total_read_mat[i, j]) * (coef1 - read_mat[i, j] / total_read_mat[i, j]) / np.sqrt((prop[i, j] * (1 - prop[i, j])))
            b_mat[i, j] = np.sqrt(total_read_mat[i, j]) * (coef2 ) / np.sqrt((prop[i, j] * (1 - prop[i, j])))
            
            # a_mat[i, j] = np.sqrt(total_read_mat[i, j]) * (coefs[1] * tag1 + coefs[5] * tag2 + coefs[3] * tag3 * tag4 - read_mat[i, j] / total_read_mat[i, j]) / np.sqrt((prop[i, j] * (1 - prop[i, j])))
            # b_mat[i, j] = np.sqrt(total_read_mat[i, j]) * (coefs[0] * tag1 + coefs[4] * tag2 + coefs[2] * tag3 * tag4 ) / np.sqrt((prop[i, j] * (1 - prop[i, j])))
    
    return a_mat, b_mat



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

def get_minor_cn_mat(df):
    n = len(np.unique(df.mutation))
    m = sum(df.mutation == np.unique(df.mutation)[0])
    res = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            index = i * m + j
            res[i, j] = df.iloc[index , ].minor_cn
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

def update_v_SCAD(index_v, pairs_mapping_inverse, p_vec, y, n, m, rho, omega, gamma):
    start_y = index_v * m
    end_y = (index_v + 1) * m
    l1, l2 = pairs_mapping_inverse[index_v]
    a_mat = a_mat_generator(l1, l2, n, m)
    temp = matmul_by_torch(a_mat, p_vec) - y[start_y: end_y] / rho
    Lambda = gamma * omega
    v = ST_vector(temp, Lambda / rho) * (np.linalg.norm(temp) <  Lambda / rho + Lambda) + \
        ST_vector(temp, 3.7 * Lambda / (2.7 * rho)) / (1 - 1/ (2.7 * rho)) * (np.linalg.norm(temp) >=  Lambda / rho + Lambda) * (np.linalg.norm(temp) <=  Lambda * 3.7) + \
            temp * (np.linalg.norm(temp) >  Lambda * 3.7)  
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

def ST(x, lam):
	val = np.abs(x) - lam
	val = np.sign(x)*(val > 0) * val
	return val

def ST_vector(x, lam):
    temp = np.linalg.norm(x)
    if temp > lam:
        return (1 - lam / temp) * x
    else:
        return np.zeros(x.shape)
    
def get_DELTA(No_mutation):
    col_id = np.append(np.array(range(int(No_mutation * (No_mutation - 1) / 2))),
                        np.array(range(int(No_mutation * (No_mutation - 1) / 2))))
    row1 = np.zeros(int(No_mutation * (No_mutation - 1) / 2))
    row2 = np.zeros(int(No_mutation * (No_mutation - 1) / 2))
    starting = 0
    for i in range(No_mutation - 1):
        row1[starting:(starting + No_mutation - i - 1)] = i
        row2[starting:(starting + No_mutation - i - 1)] = np.array(range(No_mutation))[(i + 1):]
        starting = starting + No_mutation - i - 1
    row_id = np.append(row1, row2)
    vals = np.append(np.ones(int(No_mutation * (No_mutation - 1) / 2)),
                        -np.ones(int(No_mutation * (No_mutation - 1) / 2)))
    DELTA = sp.sparse.coo_matrix((vals, (row_id, col_id)),
                                    shape=(No_mutation, int(No_mutation * (No_mutation - 1) / 2))).tocsr()
    return DELTA

def update_p(p, v, y, n, m, read_mat, total_read_mat, bb_mat, tumor_cn_mat, coef, wcut, combinations_2, pairs_mapping, rho, control_large):
    No_mutation = n * m
    theta_hat = np.reshape(read_mat / total_read_mat, [No_mutation])

    theta = np.exp(p) * np.reshape(bb_mat, [No_mutation]) / (2 + np.exp(p) * np.reshape(tumor_cn_mat, [No_mutation]))

    A = np.sqrt(np.reshape(total_read_mat, [No_mutation])) * (
                (p <= wcut[:, 0]) * coef[:, 1] + (p >= wcut[:, 1]) * coef[:, 5] + (p > wcut[:, 0]) * (
                    p < wcut[:, 1]) * coef[:, 3] - theta_hat) / np.sqrt(theta * (1 - theta))
    B = np.sqrt(np.reshape(total_read_mat, [No_mutation])) * (
                (p <= wcut[:, 0]) * coef[:, 0] + (p >= wcut[:, 1]) * coef[:, 4] + (p > wcut[:, 0]) * (
                    p < wcut[:, 1]) * coef[:, 2]) / np.sqrt(theta * (1 - theta))

    linear = rho * get_v_mat(v, y, rho, combinations_2, pairs_mapping, n, m) - (B * A)

    Minv = 1 / (B ** 2 + No_mutation * rho)
    Minv_diag = np.diag(Minv)

    trace_g = -rho * np.sum(Minv)

    Minv_outer = np.outer(Minv,Minv)
    inverted = Minv_diag - (1 / (1 + trace_g) * (-rho) * Minv_outer)
    p_new    = matmul_by_torch(inverted, linear.T)
    p_new = p_new.reshape([No_mutation])
    p_new[p_new > control_large] = control_large
    p_new[p_new < -control_large] = -control_large
    return p_new

@ray.remote(num_returns=1)
def CliPP2(df, rho, gamma, omega, n, m, max_iteration = 1000, precision=1e-2, control_large = 5):
    """    
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
    bb_mat = get_b_mat(df)
    tumor_cn_mat = get_tumor_cn_mat(df)
    linearApprox = get_linear_approximation(c_mat)
    # Initialize variables
    #12/12/2024
    phi_hat = (read_mat / (total_read_mat * c_mat))
    scale_parameter = np.max([1, np.max(phi_hat)])
    phi_hat = phi_hat / scale_parameter
    phi_hat[phi_hat > sigmoid(control_large)] = sigmoid(control_large)
    phi_hat[phi_hat < sigmoid(-control_large)] = sigmoid(-control_large)
    p = inverse_sigmoid(phi_hat)
    p[p > control_large] = control_large
    p[p < -control_large] = -control_large
    p = p.reshape([n * m])
    v = np.zeros([len(combinations_2) * m])
    for i in range(len(combinations_2)):
        pair = combinations_2[i]
        index_v = pairs_mapping[pair]
        start_v = index_v * m
        end_v = (index_v + 1) * m
        l1, l2 = pairs_mapping_inverse[index_v]
        a_mat = a_mat_generator(l1, l2, n, m)
        v[start_v: end_v] = matmul_by_torch(a_mat, p)
        
    y = np.ones([len(combinations_2) * m])
    omega = np.ones([len(combinations_2)])
    k = 0

    control_large = 5
    wcut = np.array(linearApprox[0])
    coef = np.array(linearApprox[1])
    temp = 100
    # ADMM
    while k < max_iteration and precision < temp:
        p = update_p(p, v, y, n, m, read_mat, total_read_mat, bb_mat, tumor_cn_mat, coef, wcut, combinations_2, pairs_mapping, rho, control_large)
        temp = 0
        for i in range(len(combinations_2)):
            pair = combinations_2[i]
            index_v = pairs_mapping[pair]
            start_v = index_v * m
            end_v = (index_v + 1) * m
            v[start_v: end_v] = update_v_SCAD(index_v, pairs_mapping_inverse, p, y, n, m, rho, omega[i], gamma)                
            y[start_v: end_v] = update_y(y[start_v: end_v], v[start_v: end_v], i, pairs_mapping_inverse, p, n, m, rho)
            l1, l2 = pairs_mapping_inverse[index_v]
            a_mat = a_mat_generator(l1, l2, n, m)
            temp = max(temp, np.linalg.norm(matmul_by_torch(a_mat, p) - v[start_v: end_v]))
        rho = 1.02 * rho
        k = k + 1
        # print('\r', k, ',', temp, end="")
            
    diff = np.zeros((n, n))
    class_label = -np.ones(n)
    class_label[0] = 0
    group_size = [1]
    labl = 1
    least_mut = np.ceil(0.05 * n)
    for i in range(1, n):
        for j in range(i):
            index_v = pairs_mapping[(j, i)]
            start_v = index_v * m
            end_v = (index_v + 1) * m
            diff[j, i] = np.linalg.norm(v[start_v: end_v]) if np.linalg.norm(v[start_v: end_v]) > 0.05 else 0
            diff[i, j] = diff[j, i]
    
    for i in range(1, n):
        for j in range(i):
            if diff[j, i] == 0:
                class_label[i] = class_label[j]
                group_size[int(class_label[j])] += 1
                break
        if class_label[i] == -1:
            class_label[i] = labl
            labl += 1
            group_size.append(1)

            
    # ----------------------------------------------------------
    #  PART A: Refine small clusters by reassigning to closest
    # ----------------------------------------------------------
    tmp_size = np.min(np.array(group_size)[np.array(group_size) > 0])  
    tmp_grp  = np.where(group_size == tmp_size)

    refine = False
    if tmp_size < least_mut:
        refine = True

    while refine:
        refine = False
        # Find which SNVs belong to the smallest cluster (tmp_grp[0][0])
        smallest_cluster = tmp_grp[0][0]
        tmp_col = np.where(class_label == smallest_cluster)[0]
        
        for i in range(len(tmp_col)):
            mut_idx = tmp_col[i]

            # -- Gather distance from mut_idx to all other SNVs
            if mut_idx != 0 and mut_idx != (n - 1):
                tmp_diff = np.abs(
                    np.concatenate((
                        diff[0:mut_idx, mut_idx].ravel(),
                        [100],  # placeholder for self
                        diff[mut_idx, (mut_idx+1):n].ravel()
                    ))
                )
                # Increase distances to members of tmp_col so we don't pick same cluster
                tmp_diff[tmp_col] += 100

                # Put these adjusted distances back
                diff[0:mut_idx, mut_idx] = tmp_diff[0:mut_idx]
                diff[mut_idx, (mut_idx+1):n] = tmp_diff[(mut_idx+1):n]

            elif mut_idx == 0:
                # Edge case: first row
                tmp_diff = np.append(100, diff[0, 1:n])
                tmp_diff[tmp_col] += 100
                diff[0, 1:n] = tmp_diff[1:n]

            else: 
                # Edge case: last row (mut_idx == n-1)
                tmp_diff = np.append(diff[0:(n-1), n-1], 100)
                tmp_diff[tmp_col] += 100
                diff[0:(n-1), n-1] = tmp_diff[0:(n-1)]

            # -- Pick cluster of the SNV that has the smallest distance in tmp_diff
            ind = tmp_diff.argmin()

            # Decrement the old cluster size
            old_clust = int(class_label[mut_idx])
            group_size[old_clust] -= 1

            # Reassign to the cluster of 'ind'
            new_clust = int(class_label[ind])
            class_label[mut_idx] = new_clust
            group_size[new_clust] += 1

        # Check if there's still a too-small cluster
        tmp_size = np.min(np.array(group_size)[np.array(group_size) > 0])
        tmp_grp  = np.where(group_size == tmp_size)
        refine   = (tmp_size < least_mut)
        
    # ----------------------------------------------------------
    #  PART B: Recompute cluster centroids
    # ----------------------------------------------------------
    labels = np.unique(class_label)

    phi_out = np.zeros((len(labels), m))  # shape = (#clusters x m)

    for i in range(len(labels)):
        cluster_id = labels[i]
        ind = np.where(class_label == cluster_id)[0]

        # Re-assign members to a contiguous labeling i (0..K-1)
        class_label[ind] = i

        # Weighted average (across multiple regions)
        numerator   = np.sum(phi_hat[ind, :] * total_read_mat[ind, :], axis=0)
        denominator = np.sum(total_read_mat[ind, :], axis=0)
        phi_out[i, :] = numerator / denominator
    
    
    # ----------------------------------------------------------
    #  PART C: Merge clusters with small difference in *scalar*
    # ----------------------------------------------------------
    if len(labels) > 1:
        # We'll define a single scalar "phi_scalar[i]" for each cluster i,
        # using the same coverage weighting but flattening *all* regions into 1 value.
        # This matches the single-region logic, i.e. we get "one phi" per cluster.

        def compute_cluster_scalar(phi_center):
            # For example, you might just take the mean of the m-dimensional centroid:
            # return np.mean(phi_center)
            
            # Or, to be consistent with single-region code's weighting logic,
            # we can multiply by sum over all m's coverage. But typically the centroid
            # is already coverage-weighted. So let's do a simple average:
            return np.mean(phi_center)  

        phi_scalar = np.array([compute_cluster_scalar(phi_out[i,:]) for i in range(len(labels))])

        # Sort those scalars
        sort_vals = np.sort(phi_scalar)
        phi_diff = sort_vals[1:] - sort_vals[:-1]
        min_val  = phi_diff.min()
        min_ind  = phi_diff.argmin()

        least_diff = 0.01
        while min_val < least_diff:
            # Identify the two clusters with the smallest difference
            smaller_val = sort_vals[min_ind]
            bigger_val  = sort_vals[min_ind+1]

            # Which cluster(s) have these centroid values?
            combine_ind    = np.where(phi_scalar == smaller_val)[0]
            combine_to_ind = np.where(phi_scalar == bigger_val)[0]

            # Reassign all members in 'combine_ind' to 'combine_to_ind' 
            # (just pick the first if there's more than one)
            c_from = combine_ind[0]
            c_to   = combine_to_ind[0]
            class_label[class_label == c_from] = c_to

            # Recompute the unique label set
            labels = np.unique(class_label)

            # Recompute phi_out for the new cluster set
            phi_out = np.zeros((len(labels), m))
            for idx, lbl in enumerate(labels):
                cluster_members = np.where(class_label == lbl)[0]
                # Re-map them to a contiguous cluster index 'idx'
                class_label[cluster_members] = idx

                numerator   = np.sum(phi_hat[cluster_members, :] * total_read_mat[cluster_members, :], axis=0)
                denominator = np.sum(total_read_mat[cluster_members, :], axis=0)
                phi_out[idx, :] = numerator / denominator

            if len(labels) == 1:
                break

            # Recompute the scalar representation for each cluster
            phi_scalar = np.array([compute_cluster_scalar(phi_out[i,:]) for i in range(len(labels))])

            # Sort and repeat
            sort_vals = np.sort(phi_scalar)
            phi_diff  = sort_vals[1:] - sort_vals[:-1]
            min_val   = phi_diff.min()
            min_ind   = phi_diff.argmin()
            
            
    phi_res = np.zeros((n, m))
    for k in range(len(labels)):
        # all SNVs that belong to cluster k
        idx_k = np.where(class_label == k)[0]
        phi_res[idx_k, :] = phi_out[k, :]
    
    purity = get_purity_mat(df)[0,0]
    return {'phi': phi_res, 'label': class_label, 'purity' : purity}



def preprocess_for_clipp2(snv_df, cn_df, purity, sample_id="unknown_sample",
                          valid_snvs_threshold=0, diff_cutoff=0.1):
    """
    Python version of the 'clipp1'-style preprocessing for multi-region CliPP2.
    Returns in-memory arrays/data for direct use by CliPP2.
    
    Parameters
    ----------
    snv_df : pd.DataFrame
        Must have columns: ["chromosome_index", "position", "ref_count", "alt_count"].
        One row per SNV (or per SNV-region if you are combining multiple regions).
    cn_df : pd.DataFrame
        Must have columns: [chr, start, end, minor_cn, major_cn, total_cn].
        Typically one row per CN segment.
    purity : float
        Sample-level tumor purity.
    sample_id : str
        Used for logging or error messages.
    valid_snvs_threshold : int
        Minimum required SNV count after filtering. Raises ValueError if below this.
    diff_cutoff : float
        Maximum piecewise-linear approximation error allowed for \(\theta\).

    Returns
    -------
    result_dict : dict
        {
            "snv_df_final": final filtered SNV DataFrame,
            "minor_read": 1D array of alt counts,
            "total_read": 1D array of alt+ref counts,
            "minor_count": 1D array of final minor counts,
            "total_count": 1D array of final total CN,
            "coef": 2D array of piecewise-linear coefficients (one row per SNV),
            "cutbeta": 2D array of the cut-points (one row per SNV),
            "excluded_SNVs": list of strings describing dropped SNVs,
            "purity": float
        }
    """
    ############################
    # Some small helper functions
    ############################
    def combine_reasons(chroms, positions, indices, reason):
        return [
            f"{chroms[i]}\t{positions[i]}\t{reason}"
            for i in indices
        ]
    
    def theta_func(w, bv, cv, cn, pur):
        """
        R-style: theta = (exp(w)*bv) / ((1+exp(w))*cn*(1-pur) + (1+exp(w))*cv*pur)
        """
        num = np.exp(w) * bv
        den = ((1 + np.exp(w)) * cn * (1 - pur) + (1 + np.exp(w)) * cv * pur)
        return num / den

    def linear_evaluate(x, a, b):
        return a*x + b

    def linear_approximate(bv, cv, cn, pur, diag_plot=False):
        """
        Piecewise-linear approximation of theta(w), enumerating breakpoints.
        """
        w_vals = np.arange(-5, 5.01, 0.1)
        actual = theta_func(w_vals, bv, cv, cn, pur)
        
        best_diff = float('inf')
        best_cuts = None
        best_coef = None

        for i in range(1, len(w_vals)-2):
            for j in range(i+1, len(w_vals)-1):
                w1, w2 = w_vals[i], w_vals[j]
                # 3 segments: [0..i], [i+1..j], [j+1..end]
                
                # seg1 slope/intercept
                b1 = (actual[i] - actual[0]) / (w1 - w_vals[0])
                a1 = actual[0] - w_vals[0]*b1
                # seg2 slope/intercept
                b2 = (actual[j] - actual[i]) / (w2 - w1)
                a2 = actual[i] - w1*b2
                # seg3 slope/intercept
                b3 = (actual[-1] - actual[j]) / (w_vals[-1] - w2)
                a3 = actual[-1] - w_vals[-1]*b3

                approx1 = linear_evaluate(w_vals[:i+1], b1, a1)
                approx2 = linear_evaluate(w_vals[i+1:j+1], b2, a2)
                approx3 = linear_evaluate(w_vals[j+1:], b3, a3)
                approx_val = np.concatenate((approx1, approx2, approx3))
                
                diff = np.max(np.abs(actual - approx_val))
                if diff < best_diff:
                    best_diff = diff
                    best_cuts = [w1, w2]
                    best_coef = [b1, a1, b2, a2, b3, a3]

        return {
            "w_cut": best_cuts,
            "diff": best_diff,
            "coef": best_coef
        }

    ############################
    # Start actual logic
    ############################
    dropped_reasons = []

    # 1) Filter out invalid chromosome
    mask_chr = snv_df["chromosome_index"].notna()
    drop_inds = np.where(~mask_chr)[0]
    if len(drop_inds) > 0:
        dropped_reasons += combine_reasons(
            snv_df["chromosome_index"].values,
            snv_df["position"].values,
            drop_inds,
            "Missing/invalid chromosome index"
        )
    snv_df = snv_df[mask_chr].reset_index(drop=True)
    if len(snv_df) < valid_snvs_threshold:
        raise ValueError(f"{sample_id}: only {len(snv_df)} SNVs left after invalid chromosome removal; need >= {valid_snvs_threshold}.")

    # 2) Negative or invalid read counts
    alt_arr = snv_df["alt_count"].values
    ref_arr = snv_df["ref_count"].values
    tot_arr = alt_arr + ref_arr
    mask_reads = (alt_arr >= 0) & (tot_arr >= 0)
    drop_inds = np.where(~mask_reads)[0]
    if len(drop_inds) > 0:
        dropped_reasons += combine_reasons(
            snv_df["chromosome_index"].values,
            snv_df["position"].values,
            drop_inds,
            "Negative or invalid read counts"
        )
    snv_df = snv_df[mask_reads].reset_index(drop=True)
    if len(snv_df) < valid_snvs_threshold:
        raise ValueError(f"{sample_id}: only {len(snv_df)} with valid read counts; need >= {valid_snvs_threshold}.")

    # Recalc
    alt_arr = snv_df["alt_count"].values
    ref_arr = snv_df["ref_count"].values
    tot_arr = alt_arr + ref_arr

    # 3) Match CN segment
    cn_df_ = cn_df.dropna(subset=["minor_cn"]).reset_index(drop=True)
    if len(cn_df_) == 0:
        raise ValueError(f"{sample_id}: no valid CN rows after dropping NA minor_cn.")

    def match_cn(schr, spos):
        row_ids = cn_df_.index[
            (cn_df_.iloc[:, 0] == schr) &
            (cn_df_.iloc[:, 1] <= spos) &
            (cn_df_.iloc[:, 2] >= spos)
        ]
        return row_ids[0] if len(row_ids)>0 else -1

    matched_idx = [match_cn(snv_df.loc[i,"chromosome_index"], snv_df.loc[i,"position"])
                   for i in range(len(snv_df))]
    matched_idx = np.array(matched_idx)

    mask_cn = (matched_idx >= 0)
    drop_inds = np.where(~mask_cn)[0]
    if len(drop_inds) > 0:
        dropped_reasons += combine_reasons(
            snv_df["chromosome_index"].values,
            snv_df["position"].values,
            drop_inds,
            "No matching CN segment"
        )
    snv_df = snv_df[mask_cn].reset_index(drop=True)
    matched_idx = matched_idx[mask_cn]
    if len(snv_df) < valid_snvs_threshold:
        raise ValueError(f"{sample_id}: only {len(snv_df)} with valid CN match; need >= {valid_snvs_threshold}.")

    alt_arr = snv_df["alt_count"].values
    ref_arr = snv_df["ref_count"].values
    tot_arr = alt_arr + ref_arr

    tot_cn   = cn_df_["total_cn"].values[matched_idx]
    minor_cn = cn_df_["minor_cn"].values[matched_idx]
    major_cn = cn_df_["major_cn"].values[matched_idx]
    minor_lim = np.maximum(minor_cn, major_cn)

    # 4) Multiplicity
    mult = np.round(
        (alt_arr / tot_arr / purity) * (tot_cn * purity + (1 - purity)*2)
    ).astype(int)
    # minor_count = np.minimum(mult, minor_lim)
    minor_count = np.minimum(mult, minor_lim)
    minor_count[minor_count == 0] = 1

    mask_valid_mult = (minor_count>0) & (tot_cn>0)
    drop_inds = np.where(~mask_valid_mult)[0]
    if len(drop_inds) > 0:
        dropped_reasons += combine_reasons(
            snv_df["chromosome_index"].values,
            snv_df["position"].values,
            drop_inds,
            "Invalid multiplicities (<=0)"
        )
    snv_df = snv_df[mask_valid_mult].reset_index(drop=True)
    minor_count = minor_count[mask_valid_mult]
    tot_cn      = tot_cn[mask_valid_mult]

    if len(snv_df) < valid_snvs_threshold:
        raise ValueError(f"{sample_id}: only {len(snv_df)} remain after multiplicity filtering; need >= {valid_snvs_threshold}.")

    # 5) Piecewise approximation of theta
    #    For each SNV => key (cn=2, total_cn, minor_count)
    def combo_key(tcount, mcount):
        return f"{tcount}_{mcount}"

    cache_good = {}
    cache_bad  = set()
    sample_coef = np.zeros((len(snv_df),6))
    sample_cut  = np.zeros((len(snv_df),2))
    sample_diff = np.zeros(len(snv_df))

    for i in range(len(snv_df)):
        key = combo_key(tot_cn[i], minor_count[i])
        if key in cache_good:
            # load
            sample_coef[i,:] = cache_good[key]["coef"]
            sample_cut[i,:]  = cache_good[key]["cut"]
            sample_diff[i]   = cache_good[key]["diff"]
        elif key in cache_bad:
            sample_diff[i] = 999
        else:
            # new approximation
            res = linear_approximate(
                bv = minor_count[i], 
                cv = tot_cn[i],
                cn = 2,
                pur = purity
            )
            if res["diff"] <= diff_cutoff:
                sample_coef[i,:] = res["coef"]
                sample_cut[i,:]  = res["w_cut"]
                sample_diff[i]   = res["diff"]
                cache_good[key]  = {
                    "coef": res["coef"],
                    "cut":  res["w_cut"],
                    "diff": res["diff"]
                }
            else:
                sample_diff[i] = 999
                cache_bad.add(key)

    mask_approx = (sample_diff <= diff_cutoff)
    drop_inds = np.where(~mask_approx)[0]
    if len(drop_inds)>0:
        dropped_reasons += combine_reasons(
            snv_df["chromosome_index"].values,
            snv_df["position"].values,
            drop_inds,
            "Piecewise approx error>cutoff"
        )
    snv_df = snv_df[mask_approx].reset_index(drop=True)
    sample_coef = sample_coef[mask_approx,:]
    sample_cut  = sample_cut[mask_approx,:]
    minor_count = minor_count[mask_approx]
    tot_cn      = tot_cn[mask_approx]

    if len(snv_df) < valid_snvs_threshold:
        raise ValueError(f"{sample_id}: only {len(snv_df)} remain after approx filter; need >= {valid_snvs_threshold}.")

    # Recalc alt/ref after final drop
    alt_arr = snv_df["alt_count"].values
    ref_arr = snv_df["ref_count"].values
    tot_arr = alt_arr + ref_arr

    # 6) Evaluate phi
    #   phi = 2 / ( minor_count/(alt_arr/tot_arr) - tot_cn + 2 )
    fraction = alt_arr / tot_arr
    fraction[fraction<=0] = 1e-12
    phi = 2.0 / ((minor_count/fraction) - tot_cn + 2.0)

    # keep phi <=1.5 and >0
    mask_phi = (phi>0) & (phi<=1.5)
    drop_inds = np.where(~mask_phi)[0]
    if len(drop_inds)>0:
        dropped_reasons += combine_reasons(
            snv_df["chromosome_index"].values,
            snv_df["position"].values,
            drop_inds,
            "Phi out of range (<=0 or>1.5)"
        )
    snv_df = snv_df[mask_phi].reset_index(drop=True)
    sample_coef = sample_coef[mask_phi,:]
    sample_cut  = sample_cut[mask_phi,:]
    minor_count = minor_count[mask_phi]
    tot_cn      = tot_cn[mask_phi]

    if len(snv_df) < valid_snvs_threshold:
        raise ValueError(f"{sample_id}: only {len(snv_df)} remain after phi filter; need >= {valid_snvs_threshold}.")

    # Final results
    result_dict = {
        "snv_df_final": snv_df,
        "minor_read": alt_arr[mask_phi],
        "total_read": (alt_arr+ref_arr)[mask_phi],
        "minor_count": minor_count,
        "total_count": tot_cn,
        "coef": sample_coef,
        "cutbeta": sample_cut,
        "excluded_SNVs": dropped_reasons,
        "purity": purity
    }
    return result_dict

def build_cliPP2_input(preproc_res):
    """
    Convert the dictionary from preprocess_for_clipp2(...) 
    into a DataFrame that matches the columns needed by CliPP2.
    
    For multi-region data, you would expand this by replicating each SNV 
    across multiple 'regions' or 'samples', each with distinct alt_counts, etc.
    
    Returns
    -------
    df_for_cliPP2 : pd.DataFrame
       Columns: [mutation, alt_counts, ref_counts, major_cn, minor_cn, normal_cn, tumour_purity]
    n : int
       Number of unique mutations
    m : int
       Number of samples (here 1 if single region)
    """

    snv_df_final = preproc_res["snv_df_final"]
    alt_counts   = preproc_res["minor_read"]
    tot_counts   = preproc_res["total_read"]  # alt+ref
    minor_count  = preproc_res["minor_count"] # from CN step
    total_count  = preproc_res["total_count"]
    purity       = preproc_res["purity"]
    
    n_snv = len(snv_df_final)  # one row per SNV if single region
    # For multi-region, you'd have n_snv * n_regions
    # Here we do single region => m=1
    df_for_cliPP2 = pd.DataFrame({
        "mutation": np.arange(n_snv),  # 0..n_snv-1
        "alt_counts": alt_counts,
        "ref_counts": (tot_counts - alt_counts),
        # We can define minor_cn, major_cn from (minor_count, total_count-minor_count)
        "minor_cn": minor_count,
        "major_cn": (total_count - minor_count),
        "normal_cn": 2,            # typical assumption
        "tumour_purity": purity    # same purity repeated
    })
    n = len(np.unique(df_for_cliPP2.mutation))
    m = 1
    return df_for_cliPP2, n, m


def run_preproc_and_CliPP2(
    snv_df, cn_df, purity, sample_id,
    gamma_list,
    rho = 0.8, omega = 1,
    max_iteration=1000, precision=1e-2, control_large=5,
    valid_snvs_threshold=0, diff_cutoff=0.1
):
    """
    End-to-end function:
     1) Preprocess SNVs
     2) Build a DataFrame with columns required by CliPP2
     3) Run CliPP2
    
    Parameters
    ----------
    snv_df, cn_df : pd.DataFrame
        Raw data frames for SNV & CN input
    purity : float
        Sample-level purity
    sample_id : str
        For logging
    rho, gamma, omega : float
        ADMM parameters for CliPP2
    max_iteration : int
        ...
    precision : float
        ...
    control_large : float
        ...
    valid_snvs_threshold : int
        Minimally required SNV count post-filter
    diff_cutoff : float
        Piecewise approximation error cutoff
    
    Returns
    -------
    final_result : dict
        Contains:
            "phi": final cluster membership proportions
            "label": cluster labels for each SNV
            "purity": float
            plus any other keys from the CliPP2 output
    """
    # 1) Preprocess
    preproc_res = preprocess_for_clipp2(
        snv_df, cn_df, purity, 
        sample_id=sample_id,
        valid_snvs_threshold=valid_snvs_threshold,
        diff_cutoff=diff_cutoff
    )
    
    # 2) Build input DataFrame for CliPP2
    df_for_cliPP2, n, m = build_cliPP2_input(preproc_res)
    
    # 3) Invoke your existing CliPP2 code
    #    Note that df_for_cliPP2 must have columns:
    #    "mutation", "alt_counts", "ref_counts", "major_cn", "minor_cn", "normal_cn", "tumour_purity"
    ray.shutdown()
    ray.init()
    clipp2_result = [CliPP2.remote(
        df_for_cliPP2, rho, gamma_list[i], omega,
        n, m,
        max_iteration=max_iteration,
        precision=precision,
        control_large=control_large
    ) for i in range(len(gamma_list))]
    final_result = ray.get(clipp2_result)
    ray.shutdown()

    return final_result
