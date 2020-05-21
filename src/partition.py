from random import Random

import torch
import torch.distributed as dist
from torchvision import datasets, transforms


# Dataset partitioning helper
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes, seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

# Partitioning MNIST
def partition_dataset(partition_sizes=None, train=True, custom=False):
    dataset = datasets.MNIST('./data', train=train, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    size = dist.get_world_size()
    bsz = int(128 / float(size))
    if partition_sizes is None:
        partition_sizes = [1.0 / size for _ in range(size)]
    if custom:
        if dist.get_rank() == 0:
            mask = dataset.targets < 5
        else:
            mask = dataset.targets >= 5
        indices = torch.masked_select(torch.arange(0, len(dataset)), mask) 
        partition = Partition(dataset, indices)
    else:    
        partition = DataPartitioner(dataset, partition_sizes)
        partition = partition.use(dist.get_rank())
    
    dloader = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True)
    return dloader, bsz