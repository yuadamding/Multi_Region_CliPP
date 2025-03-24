"""
approx_utils.py

Piecewise-linear approximations (theta-like functions, linear segments)
and related helper functions.
"""

import numpy as np
from .math_utils import sigmoid

def linear_evaluate(x, a, b):
    """
    Returns a*x + b.
    """
    return a * x + b

def theta(w, c):
    """
    Given w and c, computes (exp(w)*c)/(1+exp(w)).
    Essentially a logistic function scaled by c.
    """
    return (np.exp(w) * c) / (1 + np.exp(w))

def linear_approximate(c):
    """
    Builds a piecewise-linear approximation for theta(w,c) over w in [-4,4]
    with breakpoints at -1.8 and 1.8.

    Returns
    -------
    dict
        'w_cut': [w1, w2], 'coef': [b1,a1,b2,a2,b3,a3].
    """
    w = np.array([-4, -1.8, 1.8, 4])
    actual_theta = theta(w, c)

    # Segment 1
    b_1 = (actual_theta[1] - actual_theta[0]) / (w[1] - w[0])
    a_1 = actual_theta[0] - w[0] * b_1
    
    # Segment 2
    b_2 = (actual_theta[2] - actual_theta[1]) / (w[2] - w[1])
    a_2 = actual_theta[1] - w[1] * b_2
    
    # Segment 3
    b_3 = (actual_theta[3] - actual_theta[2]) / (w[3] - w[2])
    a_3 = actual_theta[2] - w[2] * b_3
    
    return {
        'w_cut': [-1.8, 1.8],
        'coef': [b_1, a_1, b_2, a_2, b_3, a_3],
    }

def get_linear_approximation(c_mat):
    """
    For each entry of c_mat, compute the piecewise-linear approximation
    of theta(w,c). Returns two lists: wcut, coef for each entry in row-major order.
    """
    n, m = c_mat.shape
    wcut, coef = [], []
    for i in range(n):
        for j in range(m):
            approx = linear_approximate(c_mat[i,j])
            wcut.append(approx['w_cut'])
            coef.append(approx['coef'])
    return wcut, coef

def get_a_and_b_mat(p_old, total_read_mat, c_mat, read_mat, n, m, linearApprox):
    """
    Constructs 'a_mat' and 'b_mat' arrays used for an approximation step.
    
    Example: This function is demonstration/placeholder; it uses made-up
    coefficients (coef1, coef2) for a piecewise approach.
    """
    p = p_old.reshape([n, m])
    cp = sigmoid(p)  # logistic transform
    prop = cp * c_mat
    a_mat = np.zeros([n, m])
    b_mat = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            if prop[i, j] >= 1:
                prop[i, j] = 1 - 1e-10
            
            # Heuristic coefficients for demonstration
            coef2 = c_mat[i, j] / 10
            coef1 = c_mat[i, j] / 2 
            
            a_mat[i, j] = (
                np.sqrt(total_read_mat[i, j])
                * (coef1 - read_mat[i, j] / total_read_mat[i, j])
                / np.sqrt(prop[i, j] * (1 - prop[i, j]))
            )
            b_mat[i, j] = (
                np.sqrt(total_read_mat[i, j])
                * coef2
                / np.sqrt(prop[i, j] * (1 - prop[i, j]))
            )
    return a_mat, b_mat