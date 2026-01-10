from .numpy_array import ndarray, get_array
from .numpy_type_constant import Int64, _dtype_promotion


# https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
def matmul(a, b): # missing args
    a, b = get_array(a), get_array(b)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul requires 2D arrays")

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"shapes {a.shape} and {b.shape} not aligned")

    m, n = a.shape[0], b.shape[1]
    p = a.shape[1]

    result = ndarray((m, n), _dtype_promotion(a.dtype, b.dtype))

    for i in range(m):
        for j in range(n):
            s = Int64(0)
            for k in range(p):
                s += a[i, k] * b[k, j]
            result[i, j] = s

    dtype = result._get_real_type()
    result._dtype = dtype
    result._verify_type(result._buffer)
    return result
