from re import L
import torch
import numpy as np

# 1. initialize tensor (if mac - cpu -> mps)
my_tensor = torch.tensor([[1,2,3],[4,5,6]],
                        dtype = torch.float32,
                        device = "mps")


print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)

# 2. Tensor from numpy array 
np_array = np.array([1,2,3,4,5])
tensor = torch.from_numpy(np_array)
print(tensor)

tensor = torch.tensor(np_array)
print(tensor)

# 3. Random constant
shape = (2,3)
rand_tensor = torch.rand(shape)
print(rand_tensor)

ones_tensor = torch.ones(shape)
print(ones_tensor)

zeros_tensor = torch.zeros(shape)
print(zeros_tensor)

# 4. indexing operation
print(rand_tensor)

take_row_0 = rand_tensor[0, :]
print(take_row_0)

take_00 = rand_tensor[:,0]
print(take_00)

# 5. Joining tensor
concat_tensor = torch.cat([ones_tensor, zeros_tensor],
                            dim = 0)

print(concat_tensor)
print(ones_tensor.shape)
print(zeros_tensor.shape)
print(concat_tensor.shape)


# 6. Operation
matrix_A = torch.rand((3,3))
column_vector = torch.rand((3,1))

print(matrix_A)
print(column_vector)

out = torch.matmul(matrix_A, column_vector)
print(out)
print(out.shape)

## element-wise operation 
ones_tensor = torch.ones((2,3))
twos_tensor = 2*torch.ones((2,3))

print(ones_tensor)
print(twos_tensor)

out = ones_tensor + twos_tensor
print(out)

## broadcasting
x1 = torch.rand((5,5))
x2 = torch.zeros((1,5))
## copy x2 from 1x5 to 5x5 --> numpy broadcasting

out = x1 * x2
print(out)

# 7. tensor reshape
x = torch.arange(9)
print(x)

x_3x3 = x.reshape(3,3)
print(x_3x3)

# GPU or cpu or apple silicon 
import platform

if torch.cuda.is_available():
    device = "cuda"
elif platform.machine() == 'arm64':
    device = "mps"
else:
    device = "cpu"
    

my_tensor = torch.tensor([[1,2,3]], dtype = torch.float32, device = device)