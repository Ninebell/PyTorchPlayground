import torch
import numpy as np

# autograd is core to all neural networks in pytorch
# if you set reuqires_grad to True, framework start to track all operations on it.
x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x+2
print(y, y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)

# how to change requires_grad
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# Gradient using Jacobian matrix
out.backward()
print(x.grad)

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)
# when you use backward in non scalar, you need to define tensor for parameter about backward
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

# when you need reuqires_grad to False, you can use with torch.no_grad() scope
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

print(x.requires_grad)
# create new tensor with same value like copy() in numpy
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())

