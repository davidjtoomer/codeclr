from typing import List

import torch
import torchmetrics

from .augmenter import Augmenter
from .encoder import Encoder
from .. import DenseGraph


class ContrastiveLearner(torch.nn.Module):
    def __init__(
            self,
            layer_sizes: List[int],
            vocab_size: int,
            mask_frac: float = 0.25,
            mask_idx: int = 0,
            augment_1: str = 'node_drop',
            augment_2: str = 'node_drop'):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.vocab_size = vocab_size
        self.mask_frac = mask_frac
        self.mask_idx = mask_idx
        self.augment_1 = augment_1
        self.augment_2 = augment_2

        self.augmenter = Augmenter(
            mask_frac=self.mask_frac,
            mask_idx=self.mask_idx,
            augment_1=self.augment_1,
            augment_2=self.augment_2)
        self.encoder = Encoder(layer_sizes, vocab_size=vocab_size)

    def forward(self, graphs: List[DenseGraph]):
        anchor_graphs, positive_graphs = self.augmenter(graphs)
        anchors = self.encoder(anchor_graphs)
        positives = self.encoder(positive_graphs)

        pos_cosine_sim = torch.nn.functional.cosine_similarity(
            anchors, positives, dim=-1)
        neg_cosine_sim = torchmetrics.functional.pairwise_cosine_similarity(
            anchors)

        numerator = torch.exp(pos_cosine_sim)
        denominator = torch.exp(neg_cosine_sim).sum(dim=-1)
        loss = -torch.log(numerator / denominator).mean()

        return loss
