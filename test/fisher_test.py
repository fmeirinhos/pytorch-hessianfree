import torch

import sys
sys.path.append("..")

from hessianfree import empirical_fisher_diagonal, empirical_fisher_matrix

B = 25
D_in = 10
D_out = 100
H = 50

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

xs = torch.rand(B, D_in)
ys = torch.rand(B, D_out)

criterion = torch.nn.MSELoss()

diag = empirical_fisher_diagonal(model, xs, ys, criterion)
matrix = empirical_fisher_matrix(model, xs, ys, criterion)

assert torch.isclose(diag, torch.diag(matrix)).any(), "Test failed"
