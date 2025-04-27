import numpy as np

def triple_product_check(u: np.array, v: np.array, w: np.array, dims):
    """
    :param dims: a tuple of length four, the first three represent the matrix dimensions
    and the last one represent the number of multiplications used
    :return: True if <u, v, w> represent a matrix multiplication algorithm of the given dimensions,
    False otherwise.
    """
    x, y, z, t = dims

    shapes = (u.shape, v.shape, w.shape)

    if u.shape[0] != t or v.shape[0] != t or w.shape[0] != t:
        return False
    if u.shape[1] != x * y or v.shape[1] != y * z or w.shape[1] != x * z:
        return False

    for i1 in range(x):
        for j1 in range(y):
            for k1 in range(z):
                for i2 in range(x):
                    for j2 in range(y):
                        for k2 in range(z):
                            res = 0
                            for r in range(t):
                                u_val = u[r, i1 * y + j1]
                                v_val = v[r, j2 * z + k1]
                                w_val = w[r, i2 * z + k2]
                                res += u_val * v_val * w_val

                            expected = int(i1 == i2 and j1 == j2 and k1 == k2)
                            if res != expected:
                                return False
    return True