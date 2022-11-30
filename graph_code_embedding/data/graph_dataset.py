import os
import random

import torch

from .. import DenseGraph


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

        self.file_names = os.listdir(data_dir)
        random.shuffle(self.file_names)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        return DenseGraph.load(os.path.join(self.data_dir, file_name))
