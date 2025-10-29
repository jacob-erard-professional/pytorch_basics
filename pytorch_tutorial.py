import torch
import numpy as np

# Messing around with tensor basics in PyTorch

x = torch.tensor([2, 2, 2])
y = torch.tensor([2, 3, 2])
print(x*y) # Element-wise multiplication
print(x@y)  # Dot product
print(x.add_(y))  # In-place addition
print(x)  # x has been modified due to in-place addition

x = torch.rand(5, 4)
print(x.view(-1, 10)) # -1 argument infers the size of that dimension

x_np = x.numpy()  # Convert to NumPy array
print(type(x_np))  # Should print <class 'numpy.ndarray'>

# Note: If a and b are running on CPU, they share the same memory, so changing
# one will change the other.

x.add_(1)
print(x_np)  # x_np reflects the change made to x

print(torch.cuda.is_available())  # Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.ones(10, device=device)  # Create a tensor on the GPU if available
y = torch.ones(10)  # Create a tensor on the CPU
y = y.to(device)  # Move y to the GPU if available
z = x + y  # Perform operation on the same device
print(z)
try:
    z.numpy()  # This will raise an error if z is on GPU
except Exception as e:
    print('tensor is running on gpu, cannot conver to numpy') 
    print(z.to("cpu").numpy())  # Move z to CPU before converting to NumPy

# By default, requires_grad is False
x = torch.ones(5, requires_grad=True)
print(x)