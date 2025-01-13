## WE USE SYMMETRIC PADDING
import numpy as np
from numba import njit, prange


@njit("Tuple((float64[:],int32[:],float64[:],int32[:],int32[:]))(int64,int64)")
def generate_kernels_layer1(input_length, num_kernels):
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)

    weights = np.zeros(lengths.sum(), dtype=np.float64)
    biases = np.zeros(num_kernels, dtype=np.float64)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    a1 = 0
    for i in range(num_kernels):
        _length = lengths[i]
        _weights = np.random.normal(0, 1, _length)
        b1 = a1 + _length
        weights[a1:b1] = _weights - _weights.mean()
        biases[i] = np.random.uniform(-1, 1)
        # Generate dilation
        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1)) // 2)
        dilation = np.int32(dilation)
        dilations[i] = dilation
        # Compute padding for 'same' output length
        paddings[i] = (dilation * (_length - 1)) // 2
        a1 = b1

    return weights, lengths, biases, dilations, paddings


@njit(fastmath=True)
def apply_kernel_layer1(X, weights, length, bias, dilation, padding):
    input_length = len(X)
    output_length = input_length  # 'Same' padding ensures output length equals input length
    X_transform = np.zeros(output_length, dtype=np.float64)
    _ppv = 0
    _max = np.NINF

    for i in range(output_length):
        _sum = bias
        for j in range(length):
            index = i - padding + j * dilation
            # Symmetric padding
            if index < 0:
                index = -index - 1
            while index >= input_length:
                index = 2 * input_length - index - 1
            _sum += weights[j] * X[index]
        X_transform[i] = _sum
        if _sum > _max:
            _max = _sum
        if _sum > 0:
            _ppv += 1

    ppv_value = _ppv / output_length
    if ppv_value > 1.0:
         print("Warning: PPV > 1.0 detected!", ppv_value, " _ppv:", _ppv, " output_length:", output_length)

    return X_transform, _ppv / output_length, _max


@njit(
    "Tuple((float64[:,:,:], float64[:,:]))(float64[:,:],Tuple((float64[::1],int32[:],float64[:],int32[:],int32[:])))",
    parallel=True,
    fastmath=True,
)
def apply_kernels_layer1(X, kernels):
    weights, lengths, biases, dilations, paddings = kernels
    num_examples, _ = X.shape
    num_kernels = len(lengths)
    X_transform = np.zeros((num_examples, num_kernels, X.shape[1]), dtype=np.float64)
    _X = np.zeros((num_examples, num_kernels * 2), dtype=np.float64)  # 2 features per kernel

    for i in prange(num_examples):
        a1 = 0  # for weights
        a2 = 0  # for features
        for j in range(num_kernels):
            b1 = a1 + lengths[j]
            b2 = a2 + 2
            x_transformed, ppv, max_val = apply_kernel_layer1(
                X[i], weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j]
            )
            X_transform[i, j] = x_transformed
            _X[i, a2] = ppv
            _X[i, a2 + 1] = max_val
            a1 = b1
            a2 = b2

    return X_transform, _X
