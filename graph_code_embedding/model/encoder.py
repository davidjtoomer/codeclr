from typing import List

import torch
from torch_geometric.nn.dense import DenseGCNConv

from .. import DenseGraph
from ..cass.cass import NodeType


class Encoder(torch.nn.Module):
    def __init__(
            self,
            layer_sizes: List[int],
            vocab_size: int = 1000,
            activation=torch.relu):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.vocab_size = vocab_size
        self.activation = activation

        self.embedding_dim = int(layer_sizes[0] / 2)

        self.node_type_embedding = torch.nn.Embedding(
            len(NodeType), self.embedding_dim)
        self.node_label_embedding = torch.nn.Embedding(
            self.vocab_size, self.embedding_dim)

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
            x = x.int()
            x = torch.cat([self.node_type_embedding(x[:, 0]),
                           self.node_label_embedding(x[:, 1])], dim=-1)
            for layer in self.layers:
                x = layer(x, adj, add_loop=False)
                x = self.activation(x)
            x = x.squeeze(0)
            pooled = torch.cat([x.mean(dim=0), x.max(dim=0)[0]], dim=-1)
            graph_embedding = self.readout(pooled)
            graph_embeddings.append(graph_embedding)
        return torch.stack(graph_embeddings)
