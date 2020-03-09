from __future__ import print_function
import torch

'''
    Create Uninitialized 5x3 matrix(tensor)
'''
x = torch.empty(5, 3)
print(x)

'''
    Create randomly initialized 5x3 matrix
'''
x = torch.rand(5, 3)
print(x)

'''
    Can create type what I want, and simply initialize zero or one
'''
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

'''
    Create tensor from data
'''
data = [5.5, 3]
x = torch.tensor(data)
print(x)

'''
    
'''
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

'''
    Make random tensor which has same size with x
'''
x = torch.randn_like(x, dtype=torch.float)
print(x)

'''
    Print x size
'''
print(x.size())

# Operation
y = torch.rand(5,3)

# operation syntax 1
print(x+y)

# operation syntax 2
print(torch.add(x, y))

# operation result
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# operation syntax 3
y.add_(x) # x to ad y // in-place addition
print(y)

# Can use standard numpy indexing
print(x[:, 1])

# resizing
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

# how to get value just only scalar
x = torch.randn(1)
print(x)
