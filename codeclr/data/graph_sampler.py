import torch


class GraphSampler(torch.utils.data.Sampler):
    def __init__(self, indices: range):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
