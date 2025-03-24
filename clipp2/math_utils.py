"""
math_utils.py

Contains core mathematical functions such as logistic/sigmoid transformations,
their derivatives, inverse, clipping arrays, etc.
"""

import numpy as np

def sigmoid(x):
    """
    Compute the logistic sigmoid of x.
    That is, sigmoid(x) = 1 / (1 + exp(-x)).

    Parameters
    ----------
    x : float or array-like
        Input value(s) for which to compute the sigmoid.

    Returns
    -------
    float or np.ndarray
        Sigmoid-transformed values in the range (0,1).
    """
    return 1 / (1 + np.exp(-x))

def adjust_array(arr, epsilon=1e-10):
    """
    Ensure the input array 'arr' has values strictly within (0,1).
    Any values <=0 are replaced with 'epsilon',
    and any values >=1 are replaced with (1-epsilon).

    Parameters
    ----------
    arr : array-like
        Input array of probabilities or values that need clipping.
    epsilon : float, optional
        The small number used to keep values inside (0,1).

    Returns
    -------
    np.ndarray
        The adjusted array with values in (epsilon, 1-epsilon).
    """
    arr = np.array(arr, copy=True)  # Ensure input is a numpy array
    arr[arr <= 0] = epsilon
    arr[arr >= 1] = 1 - epsilon
    return arr

def inverse_sigmoid(y):
    """
    Computes the inverse of the sigmoid function (i.e., the logit).
    logit(y) = log(y / (1 - y)).

    Parameters
    ----------
    y : float or np.ndarray
        Values in (0,1) (approx) to be transformed.

    Returns
    -------
    float or np.ndarray
        logit of y.
    """
    y = adjust_array(y)
    return np.log(y / (1 - y))

def sigmoid_derivative(x):
    """
    First derivative of the logistic sigmoid function.
    s'(x) = s(x)*(1 - s(x)).

    Parameters
    ----------
    x : float or np.ndarray

    Returns
    -------
    float or np.ndarray
        The derivative of sigmoid(x) at each point.
    """
    s = sigmoid(x)
    return s * (1 - s)

def sigmoid_second_derivative(x):
    """
    Second derivative of the logistic sigmoid function.
    s''(x) = s'(x)*(1 - 2*s(x)).

    Parameters
    ----------
    x : float or np.ndarray

    Returns
    -------
    float or np.ndarray
        The second derivative of the sigmoid at x.
    """
    from .math_utils import sigmoid_derivative, sigmoid
    s = sigmoid(x)
    return sigmoid_derivative(x) * (1 - 2 * s)

def ST(x, lam):
    """
    Scalar soft-thresholding. For a single value x,
    returns sign(x)*max(|x|-lam, 0).

    Parameters
    ----------
    x : float
    lam : float

    Returns
    -------
    float
        Soft-thresholded value of x.
    """
    val = abs(x) - lam
    return np.sign(x) * (val if val > 0 else 0)

def ST_vector(x, lam):
    """
    Vector soft-thresholding by L2 norm. That is,
    if ||x||>lam, scale x by (1 - lam/||x||), else zero.

    Parameters
    ----------
    x : np.ndarray
    lam : float

    Returns
    -------
    np.ndarray
        Soft-thresholded version of x.
    """
    norm_x = np.linalg.norm(x)
    if norm_x > lam:
        return (1 - lam / norm_x) * x
    else:
        return np.zeros_like(x)