import torch

# Create a sparse tensor
sparse_tensor = torch.sparse.FloatTensor(
    indices=torch.tensor([[0,1,2], [2,1,3]], dtype=torch.long),
    values=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float),
    size=(4, 4)
)

print(sparse_tensor.to_dense())

# Normalize along dimension 1 (columns)
mean = sparse_tensor.mean(dim=1)
std = sparse_tensor.std(dim=1)
normalized_tensor = (sparse_tensor - mean.unsqueeze(1)) / std.unsqueeze(1)