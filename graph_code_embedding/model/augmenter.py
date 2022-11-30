from typing import List

import torch

from .. import DenseGraph


class Augmenter(torch.nn.Module):
    def __init__(self, mask_frac: float = 0.25):
        self.mask_frac = mask_frac

    def forward(self, graphs: List[DenseGraph]):
        anchors = []
        positives = []
        for graph in graphs:
            num_nodes = graph.num_nodes
            adj = graph.adjacency_matrix

            anchor_mask = torch.rand(num_nodes) < self.mask_frac
            pos_mask = torch.rand(num_nodes) < self.mask_frac

            anchor_adj = anchor_mask.unsqueeze(
                0) * anchor_mask.unsqueeze(1) * adj
            pos_adj = pos_mask.unsqueeze(0) * pos_mask.unsqueeze(1) * adj

            anchors.append(DenseGraph(graph.node_features, anchor_adj))
            positives.append(DenseGraph(graph.node_features, pos_adj))
        return anchors, positives
