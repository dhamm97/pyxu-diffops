import time

import numpy as np
import pyxu.operator.linop.filter as pyfilt

dim_shape = (120, 40)

st_sym = pyfilt.StructureTensor(dim_shape=dim_shape, diff_method="gd", smooth_sigma=2, sigma=2, mode="symmetric")
st_const = pyfilt.StructureTensor(dim_shape=dim_shape, diff_method="gd", smooth_sigma=2, sigma=2, mode="constant")


arr = np.random.rand(1, 120, 40)

N = 1000

# -------------- TEST BOUNDARY CONDITIONS INFLUENCE -----------

# t0=time.time()
# for i in range(N):
#     st_sym(arr)
# t1=time.time()
# for i in range(N):
#     st_const(arr)
# t2=time.time()
# print(t1-t0, t2-t1)
#


# ------------- TEST SVD IMPLEMENTATION -------------


a = np.random.randn(2, 2)
A = np.dot(a, a.T)

b = np.array([A[0, 0], A[0, 1], A[1, 1]])

starr = np.hstack([b] * 4800).reshape(3, 120, 40)


# v1
t0 = time.time()
for i in range(N):
    starr_resh = np.moveaxis(starr, 0, -1)
    print(starr_resh.shape)
    starr_resh = starr_resh[..., np.array([0, 1, 1, 2])]
    u, e, _ = np.linalg.svd(starr_resh)
t1 = time.time()

print(t1 - t0)
