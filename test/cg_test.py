import numpy as np
from scipy.sparse.linalg import cg
import torch

import sys
sys.path.append("..")

from hessianfree import HessianFree

n = 50

# Matrix and vectors
A = np.random.rand(n, n)

# A must be hermitian
A = A + A.T

b = np.random.rand(n)
x0 = 1e-3 * np.ones_like(b)

# Numpy result
cg_np = cg(A=A, b=b, x0=x0, maxiter=1000)[0]

# Dummy parameters for constructor
hf = HessianFree(iter([torch.nn.Parameter()]))

A_t = torch.from_numpy(A)
b_t = torch.from_numpy(b)
x0_t = torch.from_numpy(x0)


def A_lin(vec):
    return A_t @ vec

# Torch result
cg_t = hf._CG(A=A_lin, b=b_t, x0=x0_t, max_iter=1000)[0][-1]

# Sometimes fails due to rtol
assert np.allclose(cg_np, cg_t.data.numpy(), rtol=1e-4, atol=1e-8)
