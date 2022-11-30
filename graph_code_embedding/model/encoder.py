from typing import List

import torch
from torch_geometric.nn.dense import DenseGCNConv

from .. import DenseGraph


class Encoder(torch.nn.Module):
    def __init__(self, layer_sizes: List[int]):
        self.layer_sizes = layer_sizes

        self.layers = torch.nn.ModuleList()
        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(DenseGCNConv(
                self.layer_sizes[i], self.layer_sizes[i + 1]))

        self.readout = torch.nn.Linear(
            2 * self.layer_sizes[-1], self.layer_sizes[-1])

    def forward(self, graphs: List[DenseGraph]):
        graph_embeddings = []
        for graph in graphs:
            x, adj = graph.node_features, graph.adjacency_matrix
            for layer in self.layers:
                x = layer(x, adj)
            pooled = torch.cat([x.mean(dim=0), x.max(dim=0)[0]])
            graph_embedding = self.readout(pooled)
            graph_embeddings.append(graph_embedding)
        return torch.stack(graph_embeddings)
