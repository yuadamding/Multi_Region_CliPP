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
        print('\r', k, ',', temp, end="")
            
    diff = np.zeros((n, n))
    class_label = -np.ones(n)
    class_label[0] = 0
    group_size = [1]
    labl = 1
    least_mut = 25
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

            
    # quality control
    tmp_size = np.min(np.array(group_size)[np.array(group_size) > 0])
    tmp_grp = np.where(group_size == tmp_size)
    refine = False
    if tmp_size < least_mut:
        refine = True
    while refine:
        refine = False
        tmp_col = np.where(class_label == tmp_grp[0][0])[0]
        for i in range(len(tmp_col)):
            if tmp_col[i] != 0 and tmp_col[i] != n - 1:
                tmp_diff = np.abs(np.append(np.append(diff[0:tmp_col[i], tmp_col[i]].T.ravel(), 100),
                                            diff[tmp_col[i], (tmp_col[i] + 1):n].ravel()))
                tmp_diff[tmp_col] += 100
                diff[0:tmp_col[i], tmp_col[i]] = tmp_diff[0:tmp_col[i]]
                diff[tmp_col[i], (tmp_col[i] + 1):n] = tmp_diff[(tmp_col[i] + 1):n]
            elif tmp_col[i] == 0:
                tmp_diff = np.append(100, diff[0, 1:n])
                tmp_diff[tmp_col] += 100
                diff[0, 1:n] = tmp_diff[1:n]
            else:
                tmp_diff = np.append(diff[0:(n - 1), n - 1], 100)
                tmp_diff[tmp_col] += 100
                diff[0:(n - 1), n - 1] = tmp_diff[0:(n - 1)]
            ind = tmp_diff.argmin()
            group_size[class_label.astype(np.int64, copy=False)[tmp_col[i]]] -= 1
            class_label[tmp_col[i]] = class_label[ind]
            group_size[class_label.astype(np.int64, copy=False)[tmp_col[i]]] += 1
        tmp_size = np.min(np.array(group_size)[np.array(group_size) > 0])
        tmp_grp = np.where(group_size == tmp_size)
        refine = False
        if tmp_size < least_mut:
            refine = True
    labels = np.unique(class_label)

    phi_out = np.zeros((len(labels), m))
    for i in range(len(labels)):
        ind = np.where(class_label == labels[i])[0]
        class_label[ind] = i
        phi_out[i, :] = np.sum(phi_hat[ind,: ] * total_read_mat[ind,: ]) / np.sum(total_read_mat[ind, :])
    if len(labels) > 1:
        phi_norm = np.linalg.norm(phi_out, axis=1)
        sort_phi = np.sort(phi_norm)
        indices = [np.where(phi_norm == element)[0][0] for element in sort_phi]
        phi_diff = sort_phi[1:] - sort_phi[:-1]
        min_val = phi_diff.min()
        min_ind = phi_diff.argmin()
        while min_val < 0.01:
            combine_ind = np.where(phi_out == sort_phi[indices[min_ind]])[0]
            combine_to_ind = np.where(phi_out == sort_phi[indices[min_ind] + 1])[0]
            class_label[class_label == combine_ind] = combine_to_ind
            labels = np.unique(class_label)
            phi_out = np.zeros(len(labels))
            for i in range(len(labels)):
                ind = np.where(class_label == labels[i])[0]
                class_label[ind] = i
                phi_out[i, :] = np.sum(phi_hat[ind, :] * total_read_mat[ind, :]) / np.sum(total_read_mat[ind, :])
            if len(labels) == 1:
                break
            else:
                phi_norm = np.linalg.norm(phi_out, axis=1)
                sort_phi = np.sort(phi_norm)
                indices = [np.where(phi_norm == element)[0][0] for element in sort_phi]
                phi_diff = sort_phi[1:] - sort_phi[:-1]
                min_val = phi_diff.min()
                min_ind = phi_diff.argmin()
    phi_res = np.zeros((n, m))
    for lab in range(len(phi_out)):
        phi_res[class_label == lab, :] = phi_out[lab, :]

    purity = get_purity_mat(df)[0,0]
    return {'phi': phi_res, 'label': class_label, 'purity' : purity}


def find_gamma(res):
    A_score = []
    for i in range(len(res)):
        phi_res = res[i]['phi']
        cp_norm = np.linalg.norm(phi_res, axis=1)
        A_score.append((max(cp_norm) - res[i]['purity']) / res[i]['purity'])
    A_score = np.array(A_score)
    if np.any(A_score < 0.05):
        ind1 = np.where(A_score < 0.05)
        ind2 = np.argmin(A_score[ind1])
    elif np.all(A_score > 0.01):
        ind2 = np.argmax(A_score)
    else:
        raise("Selection Failed")
    
    return ind2
        
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


def drop_snv(df):
    drop = []
    # take only non-negative counts
    read = get_read_mat(df)
    total_read = get_total_read_mat(df)
    



