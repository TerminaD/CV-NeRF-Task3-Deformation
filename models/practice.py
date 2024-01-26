import torch
from einops import rearrange
x=torch.randn((10,2))
y = rearrange(x, 'ray sample -> (ray sample)') # Assume first axis is ray
print(x)
print(y)