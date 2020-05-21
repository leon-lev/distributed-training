import os
from math import ceil

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, optim
from torch.multiprocessing import Process

from partition import partition_dataset

# Define a Network Model
class Net(nn.Module):
    """ Network architecture. """

    def __init__(self, use_batch_norm=False):
        super(Net, self).__init__()
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5, bias=False)
            self.bn1 = nn.BatchNorm2d(num_features=10)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5, bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=20)
        else:
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        if self.use_batch_norm:
            x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
            x = F.relu(F.max_pool2d(F.dropout2d(self.bn2(self.conv2(x))), 2))
        else:
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(F.dropout2d((self.conv2(x))), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Gradient averaging.
def average_gradients(model, async_op=False):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=async_op)
        param.grad.data /= size


def train_model(train_set, val_set=None,
                lr=0.01, momentum=0.5, use_batch_norm=False,
                epochs=10, async_op=False):
    
    model = Net(use_batch_norm=use_batch_norm)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_func = F.nll_loss

    train_losses, val_losses = [], []
    if val_set:
        train_accuracies, val_accuracies = [], []
    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0
        for data, target in train_set:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            train_loss += loss.item()

            ps = torch.exp(output)
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == target.view(*top_class.shape)
            train_acc += torch.mean(equals.type(torch.FloatTensor))

            loss.backward()
            average_gradients(model, async_op)
            optimizer.step()

        train_losses.append(train_loss / len(train_set))
        train_accuracies.append(train_acc / len(train_set))
        
        if val_set is None:
            print(f'Rank {dist.get_rank()}, epoch {epoch}: ',
                    f'Train Loss. {train_losses[-1]:.3f} ',
                    f'Train Acc. {train_accuracies[-1]:.3f}')
            continue

        val_loss = 0.0
        val_acc = 0.0
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for data, target in val_set:
                val_out = model(data)
                val_loss += F.nll_loss(val_out, target)
                
                ps = torch.exp(val_out)
                _, top_class = ps.topk(1, dim=1)
                equals = top_class == target.view(*top_class.shape)
                val_acc += torch.mean(equals.type(torch.FloatTensor))
                
        val_losses.append(val_loss / len(val_set))
        val_accuracies.append(val_acc / len(val_set))
        
        print(f'Rank {dist.get_rank()}, epoch {epoch}: ',
                f'Train Loss {train_losses[-1]:.3f} ',
                f'Train Acc. {train_accuracies[-1]:.3f}',
                f' --|-- Val. Loss {val_losses[-1]:.3f}',
                f'Val. Acc. {val_accuracies[-1]:.3f}')

    if val_set:
        return train_losses, train_accuracies, val_losses, val_accuracies
    else:
        return train_losses, train_accuracies


def run(rank, size, partition_sizes, custom_partition=False, params=None):
    torch.manual_seed(1234)
    train_set, _ = partition_dataset(partition_sizes, train=True, custom=custom_partition)
    val_set, _ = partition_dataset(partition_sizes, train=False)
    train_model(train_set, val_set, **params)


def init_process(fn, rank, size, *args, backend='gloo'):
    # Initialize the distributed environment.
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, *args)


if __name__ == "__main__":
    import sys
    import yaml

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = r'config.yaml'

    try:
        with open(r'config.yaml') as file:
            config_dict = yaml.load(file, Loader=yaml.SafeLoader)
    except:
        print(f'Could not open configuration file {config_file}. Aborting.')
        exit(1)

    size = config_dict['size']
    partition_sizes = config_dict['partition_sizes']
    custom_partition = config_dict['custom_partition']
    params = config_dict['params']

    processes = []
    for rank in range(size):
        p = Process(target=init_process, 
                    args=(run, rank, size, partition_sizes,
                          custom_partition, params))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()