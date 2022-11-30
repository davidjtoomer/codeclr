import torch

from .graph_dataset import GraphDataset
from .graph_sampler import GraphSampler


def identity(x):
    return x


def train_val_test_split(
        data_dir: str,
        train_frac: float = 0.8,
        batch_size: int = 1):

    dataset = GraphDataset(data_dir)
    n = len(dataset)
    train_size = int(train_frac * n)
    val_size = (n - train_size) // 2
    test_size = n - train_size - val_size

    train_indices = range(train_size)
    val_indices = range(train_size, train_size + val_size)
    test_indices = range(train_size + val_size, n)

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=GraphSampler(train_indices),
        collate_fn=identity,
        drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=GraphSampler(val_indices),
        collate_fn=identity,
        drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=GraphSampler(test_indices),
        collate_fn=identity,
        drop_last=True)
    return train_dataloader, val_dataloader, test_dataloader
