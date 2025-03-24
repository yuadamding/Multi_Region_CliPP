"""
matrix_ops.py

Matrix-building utilities, Kronecker products for pairwise difference
construction, and PyTorch-based linear algebra helpers.
"""

import numpy as np
import torch

def a_mat_generator(v1, v2, n, m):
    """
    Generates the matrix A_l that encodes the difference
    between SNV v1 and SNV v2 across m samples.

    Parameters
    ----------
    v1, v2 : int
        Indices of the SNVs.
    n : int
        Total number of SNVs.
    m : int
        Number of samples.

    Returns
    -------
    np.ndarray
        A matrix of shape (m, n*m) that is the Kronecker product of 
        a difference vector [0,...,1,...,-1,...,0] with (m x m) identity.
    """
    if n < 2 or m < 1:
        raise ValueError("The number of SNVs or number of samples is invalid.")
    
    temp1 = np.zeros(n)
    temp2 = np.zeros(n)
    temp1[v1] = 1
    temp2[v2] = 1
    a_mat = np.kron(temp1 - temp2, np.diag(np.ones(m)))
    return a_mat

def a_trans_a_mat_generator_quick(n, m):
    """
    Fast version to generate Σ_{l in Ω}(A_l^T A_l).

    Parameters
    ----------
    n, m : int
        #SNVs, #samples.

    Returns
    -------
    np.ndarray
        Shape (n*m, n*m), capturing all pairwise differences in a block structure.
    """
    init_line = [0 for _ in range(n * m)]
    init_line[0] = n - 1
    for i in range(1, n * m):
        if i % m == 0:
            init_line[i] = -1
    res = np.zeros([n * m, n * m])
    for i in range(n * m):
        # rotate the init_line
        res[i, :] = init_line[(n*m - i):] + init_line[:(n*m - i)]
    return res

def matmul_by_torch(amat, bmat):
    """
    Multiply two numpy arrays using PyTorch, returning a numpy array result.

    Parameters
    ----------
    amat : np.ndarray
    bmat : np.ndarray

    Returns
    -------
    np.ndarray
        The matrix multiplication product of amat x bmat.
    """
    amat_t = torch.from_numpy(amat)
    bmat_t = torch.from_numpy(bmat)
    res_mat = torch.matmul(amat_t, bmat_t)
    return res_mat.numpy()

def mat_inverse_by_torch(mat):
    """
    Invert a matrix using PyTorch, returning a numpy array.

    Parameters
    ----------
    mat : np.ndarray

    Returns
    -------
    np.ndarray
        Inverse of the input matrix.
    """
    mat_t = torch.from_numpy(mat)
    inv_t = torch.inverse(mat_t)
    return inv_t.numpy()

def sum_by_torch(mat):
    """
    Computes the sum of all elements in 'mat' using PyTorch,
    then returns it as a Python float or numpy scalar.

    Parameters
    ----------
    mat : np.ndarray

    Returns
    -------
    float
        The sum of all elements in 'mat'.
    """
    mat_t = torch.from_numpy(mat)
    return torch.sum(mat_t).item()