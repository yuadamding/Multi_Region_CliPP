"""
df_accessors.py

Provides utilities to extract/calculate relevant matrices (copy number,
reads, purity, etc.) from a pandas DataFrame in the format expected by CliPP2.
"""

import numpy as np
import pandas as pd

# These might use from .matrix_ops or .math_utils if needed.

def get_major_cn(df):
    """
    Extracts the major copy number (major_cn) from df
    into an (n x m) array, where n=#unique SNVs and m=#samples.
    """
    n = len(np.unique(df.mutation))
    m = sum(df.mutation == np.unique(df.mutation)[0])
    res = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            index = i * m + j
            res[i, j] = df.iloc[index, :].major_cn
    return res

def get_minor_cn_mat(df):
    """
    Builds an (n,m) array of minor_cn from df.
    """
    n = len(np.unique(df.mutation))
    m = sum(df.mutation == np.unique(df.mutation)[0])
    res = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            index = i * m + j
            res[i, j] = df.iloc[index, :].minor_cn
    return res

def get_tumor_cn_mat(df):
    """
    Builds an (n,m) array of total tumor CN (major+minor).
    """
    n = len(np.unique(df.mutation))
    m = sum(df.mutation == np.unique(df.mutation)[0])
    res = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            index = i * m + j
            res[i, j] = df.iloc[index, :].major_cn + df.iloc[index, :].minor_cn
    return res

def get_normal_cn_mat(df):
    """
    Builds an (n,m) array of normal_cn.
    """
    n = len(np.unique(df.mutation))
    m = sum(df.mutation == np.unique(df.mutation)[0])
    res = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            index = i * m + j
            res[i, j] = df.iloc[index, :].normal_cn
    return res

def get_purity_mat(df):
    """
    Builds an (n,m) array of tumour_purity from df.
    """
    n = len(np.unique(df.mutation))
    m = sum(df.mutation == np.unique(df.mutation)[0])
    res = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            index = i * m + j
            res[i, j] = df.iloc[index, :].tumour_purity
    return res

def get_read_mat(df):
    """
    Builds an (n,m) array of alt_counts from df.
    """
    n = len(np.unique(df.mutation))
    m = sum(df.mutation == np.unique(df.mutation)[0])
    res = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            index = i * m + j
            res[i, j] = df.iloc[index, :].alt_counts
    return res

def get_total_read_mat(df):
    """
    Builds an (n,m) array of total reads (alt+ref) from df.
    """
    n = len(np.unique(df.mutation))
    m = sum(df.mutation == np.unique(df.mutation)[0])
    res = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            index = i * m + j
            res[i, j] = df.iloc[index, :].alt_counts + df.iloc[index, :].ref_counts
    return res

def get_b_mat(df):
    """
    Approximate b (multiplicity) from read data and copy number.
    Clamps the rounding at major_cn.
    """
    from .df_accessors import (
        get_tumor_cn_mat,
        get_normal_cn_mat,
        get_purity_mat,
        get_major_cn,
        get_read_mat,
        get_total_read_mat
    )
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
            temp = (
                read_mat[i, j]
                * (purity_mat[i, j]*tumor_cn_mat[i, j] + (1 - purity_mat[i, j])*normal_cn_mat[i, j])
            ) / (total_read_mat[i, j]*purity_mat[i, j])
            temp_rounded = 1 if round(temp) == 0 else round(temp)
            res[i, j] = min(temp_rounded, major_cn_mat[i, j])
    return res

def get_c_mat(df):
    """
    Computes c = b / [ (1 - purity)*normal_cn + purity*tumor_cn ],
    returning an (n,m) matrix.
    """
    from .df_accessors import (
        get_tumor_cn_mat,
        get_normal_cn_mat,
        get_purity_mat,
        get_b_mat
    )
    tumor_cn_mat = get_tumor_cn_mat(df)
    normal_cn_mat = get_normal_cn_mat(df)
    purity_mat = get_purity_mat(df)
    b_mat = get_b_mat(df)
    c_mat = b_mat / ((1 - purity_mat)*normal_cn_mat + purity_mat * tumor_cn_mat)
    return c_mat