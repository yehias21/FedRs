import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10


def load_datasets(num_clients: int):
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("../clients/", train=True, download=True, transform=transform)
    testset = CIFAR10("../clients/", train=False, download=True, transform=transform)

    # Split training set into `num_clients` partitions to simulate different local datasets
    datasets = []
    for i in range(num_clients):
        # Randomly sample a set of classes from the list of available classes
        classes = trainset.classes
        partition_classes = set(np.random.choice(classes, size=np.random.randint(1, len(classes)), replace=False))
        partition_class_indices = [trainset.class_to_idx[c] for c in partition_classes]
        indices = np.where(np.isin(trainset.targets, partition_class_indices))[0]
        print("Client {}: {} samples, classes: {}".format(i, len(indices), partition_classes))

        # if len(indices) == 0:
        #     continue  # Skip this partition if it has no samples

        partition = Subset(trainset, indices)
        datasets.append(partition)

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32))

    testloader = DataLoader(testset, batch_size=32)

    return trainloaders, valloaders, testloader


def load_data():
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(root="./src/core/clients/", train=True, download=True, transform=transform)
    testset = CIFAR10(root="./src/core/clients/", train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainloader, testloader, num_examples
