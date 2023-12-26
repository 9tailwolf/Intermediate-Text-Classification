import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self,input_layer,hidden_layer,output_layer):
        super().__init__()
        self.conv1 = GCNConv(input_layer, hidden_layer)
        self.conv2 = GCNConv(hidden_layer, output_layer)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)