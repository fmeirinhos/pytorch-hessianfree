import torch

import sys
sys.path.append("..")

from hessianfree import HessianFree, empirical_fisher_diagonal

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


def M():  # preconditioner
    return empirical_fisher_diagonal(model, x, y, criterion)


optimizer = HessianFree(model.parameters(), use_gnm=True, verbose=True)

for i in range(20):
    print("Epoch {}".format(i))
    optimizer.zero_grad()
    optimizer.step(closure, M=None)

print("Target data\t {}".format(y))
print("Predicted\t {}".format(model(x)))
