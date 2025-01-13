def gini_index_3d(X):
    """
    Compute the Gini Index along the last axis of a 3D array.

    Parameters
    ----------
    X : numpy.ndarray
        Input array of shape (A, B, T) where T is the length of the time series.

    Returns
    -------
    GI : numpy.ndarray
        The Gini Index for each (A, B) slice, resulting in an array of shape (A, B).
    """
    # X is (A, B, T)
    A, B, T = X.shape

    # 1. Take absolute values
    abs_X = np.abs(X)

    # 2. Sort along the last axis (time series axis)
    sorted_X = np.sort(abs_X, axis=2)

    # 3. Compute the L1 norm along the last axis
    l1_norm = np.sum(sorted_X, axis=2, keepdims=True)

    # To avoid division by zero if a time series is all zeros:
    # If l1_norm is zero, the Gini index is not well-defined.
    # We can either leave as NaN or define GI=0. Here we choose GI=0 when l1=0.
    l1_norm_safe = np.where(l1_norm == 0, 1, l1_norm)

    # 4. Construct an index array k for weighting
    # shape: (1, 1, T) will broadcast to (A, B, T)
    k = np.arange(1, T+1)[None, None, :]

    # 5. Compute the summation:
    # sum_k ( (|f[k]|/||f||_1) * ((T - k + 0.5) / T ) ) along axis=2
    summation = np.sum((sorted_X / l1_norm_safe) * ((T - k + 0.5) / T), axis=2)

    # 6. Compute Gini Index along axis=2:
    # GI = 1 - 2 * summation
    GI = 1 - 2 * summation

    # If we replaced l1_norm with a safe version, correct GI where l1_norm was 0:
    # If no variation, GI can be considered 0.
    GI[l1_norm.squeeze(axis=2) == 0] = 0.0

    return GI
