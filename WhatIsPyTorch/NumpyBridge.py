from __future__ import print_function
import torch
import numpy as np
# Convert torch to numpy
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

# Share memory with tensor and numpy

a.add_(1)
# if you use b = b + 1 , can not share with tensor
b[:] = b[:]+1
print(a)
print(b)

# numpy array to tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

x = torch.ones(1)

if torch.cuda.is_available():
    device = torch.device('cuda')
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to('cpu', torch.double))
