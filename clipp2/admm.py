"""
admm.py

Core ADMM routines for the CliPP2 approach:
- Log-likelihood, objective function, gradient, 
- update steps for p, v, y,
- SCAD-based updates, clustering demonstration, etc.
"""

import numpy as np
import scipy as sp
from .math_utils import (
    sigmoid, sigmoid_derivative, ST_vector
)
from .matrix_ops import (
    a_mat_generator,
    matmul_by_torch,
    a_trans_a_mat_generator_quick,
    sum_by_torch
)

def get_loglikelihood(p, c_mat, read_mat, total_read_mat):
    """
    Computes the log-likelihood term under a Bernoulli model,
    with predicted prob = sigmoid(p)*c_mat (clamped to <=1 in practice).
    """
    cp = sigmoid(p)
    prop = cp * c_mat
    return sum_by_torch(
        read_mat * np.log(prop) + (total_read_mat - read_mat) * np.log(1 - prop)
    )

def get_v_mat(v, y, rho, combinations, pairs_mapping, n, m):
    """
    Builds Σ_l A_l^T [v_l + y_l/rho], used in p-update steps.
    """
    from .matrix_ops import a_mat_generator, matmul_by_torch
    v_tilde = v + y / rho
    res = np.zeros([n * m])
    for i in range(len(combinations)):
        pair = combinations[i]
        index_v = pairs_mapping[pair]
        start_v = index_v * m 
        end_v = (index_v + 1) * m
        a_mat = a_mat_generator(pair[0], pair[1], n, m)
        res += matmul_by_torch(a_mat.T, v_tilde[start_v: end_v])
    return res

def get_objective_function(p, v, y, rho, combinations, pairs_mapping,
                           c_mat, read_mat, total_read_mat, n, m, gamma, omega):
    """
    The full objective:
    = -loglikelihood + sum_l [ (rho/2)*||A_l p - v_l||^2 + y_l*(A_l p - v_l)
                              + gamma*omega[i]*||v_l||_2 ].
    """
    from .matrix_ops import a_mat_generator, matmul_by_torch
    res = -get_loglikelihood(p.reshape([n, m]), c_mat, read_mat, total_read_mat)
    for i, pair in enumerate(combinations):
        index_v = pairs_mapping[pair]
        start_v = index_v * m 
        end_v   = (index_v + 1) * m
        a_mat   = a_mat_generator(pair[0], pair[1], n, m)
        temp    = v[start_v: end_v] - matmul_by_torch(a_mat, p)
        res    += 0.5 * rho * matmul_by_torch(temp.T, temp)
        res    += np.dot(y[start_v: end_v], temp)  # y^T temp
        # L2 norm of v-block
        norm_v  = np.sqrt(np.dot(v[start_v: end_v], v[start_v: end_v]))
        res    += gamma * omega[i] * norm_v
    return res

def get_grad_of_objective_function(v, y, rho, combinations, pairs_mapping,
                                   c_mat, read_mat, total_read_mat, n, m):
    """
    Returns a closure that computes the gradient of the objective
    w.r.t p for ADMM-based optimization.
    """
    from .matrix_ops import matmul_by_torch, a_trans_a_mat_generator_quick
    from .math_utils import sigmoid_derivative, sigmoid

    v_mat = rho * get_v_mat(v, y, rho, combinations, pairs_mapping, n, m)
    a_mat = rho * a_trans_a_mat_generator_quick(n, m)

    def grad_of_objective_function(p):
        p_2d = p.reshape([n, m])
        cp_prime = sigmoid_derivative(p_2d)
        cp = sigmoid(p_2d)
        prop = cp * c_mat
        # gradient of -loglikelihood
        grad_logl = cp_prime * (read_mat - total_read_mat * prop) / (cp * (1 - prop))
        grad_logl = -grad_logl.reshape([n*m])  # negative sign for the objective

        # Then add the penalty term
        #   + (rho * A^T A) p - rho * (v + y/rho)
        #   = a_mat * p - v_mat
        penalty_part = matmul_by_torch(a_mat, p) - v_mat
        return grad_logl + penalty_part

    return grad_of_objective_function

def get_objective_function_p(v, y, rho, combinations, pairs_mapping,
                             c_mat, read_mat, total_read_mat, n, m):
    """
    Builds an objective function in terms of p only,
    ignoring direct penalty on v for local solvers.
    """
    from .matrix_ops import a_mat_generator, matmul_by_torch
    from .math_utils import sigmoid

    v_tilde = v + y / rho
    def objective_function_p(p):
        p_2d = p.reshape([n, m])
        cp = sigmoid(p_2d)
        prop = cp * c_mat
        # -loglikelihood
        res = -np.sum(read_mat * np.log(prop) + (total_read_mat - read_mat)*np.log(1 - prop))

        # penalty
        for i, pair in enumerate(combinations):
            index_v = pairs_mapping[pair]
            start_v = index_v * m 
            end_v   = (index_v + 1) * m
            a_mat   = a_mat_generator(pair[0], pair[1], n, m)
            temp    = v_tilde[start_v: end_v] - matmul_by_torch(a_mat, p)
            res    += 0.5 * rho * np.dot(temp, temp)
        return res
    return objective_function_p

def update_v(index_v, pairs_mapping_inverse, p_vec, y, n, m, rho, omega, gamma):
    """
    L1 update for v in ADMM. 
    Uses soft thresholding on ||v||_2 with parameter gamma*omega/rho.
    """
    from .matrix_ops import a_mat_generator, matmul_by_torch
    start_y = index_v * m
    end_y   = (index_v + 1)*m
    l1, l2  = pairs_mapping_inverse[index_v]
    a_mat   = a_mat_generator(l1, l2, n, m)
    temp    = matmul_by_torch(a_mat, p_vec) - y[start_y: end_y] / rho
    norm_t  = np.linalg.norm(temp)
    threshold = gamma*omega / rho
    if norm_t > threshold:
        v_block = (1 - threshold/norm_t) * temp
    else:
        v_block = np.zeros(m)
    return v_block

def update_v_SCAD(index_v, pairs_mapping_inverse, p_vec, y, n, m, rho, omega, gamma):
    """
    SCAD-based update for v in ADMM, with piecewise thresholding logic.
    (Implementation here is schematic; adapt as needed.)
    """
    from .matrix_ops import a_mat_generator, matmul_by_torch
    from .math_utils import ST_vector
    start_y = index_v * m
    end_y   = (index_v + 1) * m
    l1, l2  = pairs_mapping_inverse[index_v]
    a_mat   = a_mat_generator(l1, l2, n, m)
    temp    = matmul_by_torch(a_mat, p_vec) - y[start_y: end_y] / rho

    # This is a rough placeholder for SCAD. Typically SCAD can be done
    # via iterative reweighted methods or subgradient approach.
    # A direct closed-form SCAD for vector norms is more intricate.
    # 
    # For demonstration, we do a simple L2 soft-threshold:
    Lam = gamma * omega
    v_block = ST_vector(temp, Lam / rho)
    return v_block

def update_y(y_block, v_block, index_v, pairs_mapping_inverse, p_vec, n, m, rho):
    """
    Dual variable update: y_l <- y_l + rho*(v_l - A_l p).
    """
    from .matrix_ops import a_mat_generator, matmul_by_torch
    l1, l2 = pairs_mapping_inverse[index_v]
    a_mat  = a_mat_generator(l1, l2, n, m)
    return y_block + rho * (v_block - matmul_by_torch(a_mat, p_vec))

def dis_cluster(p, v, n, m, combinations, pairs_mapping, gamma):
    """
    Demonstration function for clustering based on v differences:
    If ||v_l|| < threshold, unify the SNVs in the same cluster.
    """
    from .matrix_ops import matmul_by_torch
    dic = {i: i for i in range(n)}
    res = np.zeros([n, n])
    for i, pair in enumerate(combinations):
        index_v = pairs_mapping[pair]
        start_v = index_v*m
        end_v   = (index_v+1)*m
        v_index = v[start_v: end_v]
        temp = np.linalg.norm(v_index)/np.sqrt(m)
        if temp < 1e-2:
            res[pair] = 1
    for i in range(n - 1):
        for j in range(i + 1, n):
            if res[i, j] == 1:
                dic[j] = dic[i]
    print(f"Gamma: {gamma}, clusters: {dic.values()}")
    return dic