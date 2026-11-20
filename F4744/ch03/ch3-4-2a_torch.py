import torch

a = torch.Tensor([[1, 2], [3, 4]])
print("a=")
print(a)
s = torch.Tensor([[5, 6], [7, 8]])
print("s=")
print(s)

b = torch.mm(a, s)       
print("torch.mm(a, s)=")
print(b)  

b = torch.matmul(a, s)       
print("torch.matmul(a, s)=")
print(b)  