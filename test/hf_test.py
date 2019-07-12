import torch

import sys
sys.path.append("..")

from hessianfree import (HessianFree,
                         empirical_fisher_diagonal,
                         empirical_fisher_matrix)

x = torch.Tensor([[0.333, 1]])
y = torch.Tensor([[0.4, 0.2]])

model = torch.nn.Sequential(
    torch.nn.Linear(2, 1),
    torch.nn.Tanh(),
    torch.nn.Linear(1, 2))
criterion = torch.nn.MSELoss()


def closure():
    z = model(x)
    loss = criterion(z, y)
    loss.backward(create_graph=True)
    return loss, z


# def M_inv():  # inverse preconditioner
#     return empirical_fisher_diagonal(model, x, y, criterion)
# def M_inv():  # inverse preconditioner
#     return empirical_fisher_matrix(model, x, y, criterion)

optimizer = HessianFree(model.parameters(), use_gnm=True, verbose=True)

for i in range(20):
    print("Epoch {}".format(i))
    optimizer.zero_grad()
    optimizer.step(closure, M_inv=None)

print("Target data\t {}".format(y))
print("Predicted\t {}".format(model(x)))
