import torch
from torch_geometric.nn import GATConv

torch.manual_seed(2023)

# Create a GAT layer
gat_layer = GATConv(in_channels=5, out_channels=10, heads=2, concat=True)

# Create a sample graph
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
x = torch.randn(3, 5)

# Pass the graph through the GAT layer and get attention scores
out, attn = gat_layer(x, edge_index, return_attention_weights=True)

print(attn)