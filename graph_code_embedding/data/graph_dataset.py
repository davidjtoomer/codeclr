import torch

from ..cass import CassConfig

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, benchmark: int, config: CassConfig):
        self.data_dir = data_dir
        self.benchmark = benchmark
        self.config = config

        self.benchmark_data_dir = os.path.join(data_dir, f'Project_CodeNet_C++{self.benchmark}')
        self.num_files = 0
        for directory in os.listdir(self.benchmark_data_dir):
            if os.path.isdir(os.path.join(self.benchmark_data_dir, directory)):
                self.num_files += len(os.listdir(os.path.join(self.benchmark_data_dir, directory)))

    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        return # TODO: convert from index to location in file system and load the graph.
