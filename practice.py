import torch

tensor = torch.tensor([1, 2, 3, 4, 5])
indices = torch.tensor([1, 3, 4])

selected_values = torch.take(tensor, indices)
print(selected_values)
