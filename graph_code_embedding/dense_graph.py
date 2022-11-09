from typing import List

import torch

class DenseGraph:
    def __init__(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> None:
        self.node_features = node_features
        self.adjacency_matrix = adjacency_matrix

    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[0]

    @property
    def num_edges(self) -> int:
        return self.adjacency_matrix.sum().item() / 2

    def save(self, path: str) -> None:
        torch.save({'node_features': self.node_features, 'adjacency_matrix': self.adjacency_matrix}, path)

    @staticmethod
    def load(path: str) -> 'DenseGraph':
        data = torch.load(path)
        return DenseGraph(node_features=data['node_features'], adjacency_matrix=data['adjacency_matrix'])
