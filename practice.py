import torch

def create_tensor_with_constant_value(n, x):
    # Create a tensor with n rows, each filled with the value x
    result_tensor = torch.full((n,), x)

    return result_tensor

# Example: Create a tensor with 5 rows, each filled with the value 3.0
n_rows = 5
constant_value = 3.0
result_tensor = create_tensor_with_constant_value(n_rows, constant_value)

print(result_tensor)
