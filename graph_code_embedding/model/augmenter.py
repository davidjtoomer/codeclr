import random
from typing import List

import torch

from .. import DenseGraph
from ..cass.cass import NodeType


class Augmenter(torch.nn.Module):
    def __init__(
        self,
        mask_frac: float = 0.25,
        mask_idx: int = 0,
        augment_1: str = 'node_mask',
        augment_2: str = 'node_mask'
    ):
        super().__init__()
        self.mask_frac = mask_frac
        self.mask_idx = mask_idx

        if augment_1 == 'identity':
            self.augment_1 = Identity()
        elif augment_1 == 'node_mask':
            self.augment_1 = NodeMasker(mask_frac=mask_frac, mask_idx=mask_idx)
        elif augment_1 == 'node_drop':
            self.augment_1 = NodeDropper(drop_frac=mask_frac)
        elif augment_1 == 'subtree_mask':
            self.augment_1 = SubtreeMasker(
                mask_frac=mask_frac, mask_idx=mask_idx)

        if augment_2 == 'identity':
            self.augment_2 = Identity()
        elif augment_2 == 'node_mask':
            self.augment_2 = NodeMasker(mask_frac=mask_frac, mask_idx=mask_idx)
        elif augment_2 == 'node_drop':
            self.augment_2 = NodeDropper(drop_frac=mask_frac)
        elif augment_2 == 'subtree_mask':
            self.augment_2 = SubtreeMasker(
                mask_frac=mask_frac, mask_idx=mask_idx)

    def forward(self, graphs: List[DenseGraph]):
        augment_1 = self.augment_1(graphs)
        augment_2 = self.augment_2(graphs)
        return augment_1, augment_2


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graphs: List[DenseGraph]):
        return graphs


class NodeMasker(torch.nn.Module):
    def __init__(self, mask_frac: float = 0.25, mask_idx: int = 0):
        super().__init__()
        self.mask_frac = mask_frac
        self.mask_idx = mask_idx

    def forward(self, graphs: List[DenseGraph]):
        augments = []
        for graph in graphs:
            num_nodes = graph.num_nodes
            node_features = graph.node_features.clone()

            num_nodes_to_mask = int(num_nodes * self.mask_frac)
            nodes_to_mask = torch.randperm(num_nodes)[:num_nodes_to_mask]
            node_features[:, 0][nodes_to_mask] = NodeType.Mask.value
            node_features[:, 1][nodes_to_mask] = self.mask_idx

            augments.append(DenseGraph(node_features, graph.adjacency_matrix))
        return augments


class NodeDropper(torch.nn.Module):
    def __init__(self, drop_frac: float = 0.25):
        super().__init__()
        self.drop_frac = drop_frac

    def forward(self, graphs: List[DenseGraph]):
        augments = []
        for graph in graphs:
            num_nodes = graph.num_nodes
            node_features = graph.node_features.clone()
            adj = graph.adjacency_matrix.clone()

            num_nodes_to_drop = int(num_nodes * self.drop_frac)
            nodes_to_drop = torch.randperm(num_nodes)[:num_nodes_to_drop]
            node_mask = torch.ones(num_nodes, dtype=torch.bool)
            node_mask[nodes_to_drop] = False

            node_features = node_features * node_mask.unsqueeze(-1)
            adj = adj * node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
            augments.append(DenseGraph(node_features, adj))
        return augments


class SubtreeMasker(torch.nn.Module):
    def __init__(self, mask_frac: float = 0.25, mask_idx: int = 0):
        super().__init__()
        self.mask_frac = mask_frac
        self.mask_idx = mask_idx

    def forward(self, graphs: List[DenseGraph]):
        augments = []
        for graph in graphs:
            num_nodes = graph.num_nodes
            node_features = graph.node_features.clone()
            adj = graph.adjacency_matrix

            num_nodes_to_mask = int(num_nodes * self.mask_frac)
            node_mask = torch.ones(num_nodes, dtype=torch.bool)

            root = random.randint(0, num_nodes - 1)
            node_mask[root] = False
            bfs_queue = [root]
            seen = set()
            while node_mask.sum().item() < num_nodes_to_mask and len(bfs_queue) > 0:
                node = bfs_queue.pop(0)
                seen.add(node)
                neighbors = adj[node].nonzero().squeeze()
                for neighbor in neighbors:
                    if neighbor not in seen:
                        bfs_queue.append(neighbor)
                        node_mask[neighbor] = False

            node_features[:, 0][node_mask] = NodeType.Mask.value
            node_features[:, 1][node_mask] = self.mask_idx

            augments.append(DenseGraph(node_features, graph.adjacency_matrix))

        return augments
