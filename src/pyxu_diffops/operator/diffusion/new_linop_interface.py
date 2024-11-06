import numpy as np
import pyxu.abc as pxa
import pyxu.operator.linop.diff as pydiff
import scipy.sparse as sp

dim_shape = (120, 40)
codim_shape = dim_shape

Dx = -np.diag(np.ones(dim_shape[0])) + np.diag(np.ones(dim_shape[0] - 1), 1)
Dx[-1, -1] = 0  # symmetric boundary conditions, no flux
Dy = -np.diag(np.ones(dim_shape[1])) + np.diag(np.ones(dim_shape[1] - 1), 1)
Dy[-1, -1] = 0  # symmetric boundary conditions, no flux
# define gradient matrix
D = np.vstack((np.kron(Dx, np.eye(dim_shape[1])), np.kron(np.eye(dim_shape[0]), Dy)))
L = D.T @ D
Lsp = sp.csr_matrix(L)

gradient = pydiff.Gradient(dim_shape=(1, *dim_shape), directions=(1, 2))


arr = np.random.rand(1, 120, 40)

# s=time.time()
# for i in range(1000): # best
#     b = arr.reshape(-1,4800)
#     f=Lsp.dot(b.T).T
#     out = f.reshape(-1,4800)
# e=time.time()
# for i in range(1000): # slow
#     b = arr.reshape(-1, 4800)
#     temp=D.dot(b.T)
#     out = D.T.dot(temp)
# ef=time.time()
#
# print(e-s)
# print(ef-e)

arrop = pxa.LinOp.from_array(Lsp)

arr.shape

# Q = pxa.LinOp.from_array(A=Lsp)
