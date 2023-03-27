from collections import OrderedDict

import flwr as fl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import numpy as np

from src.core.model.testing_model import Net
# from src.core.servers.server import trainloaders, valloaders


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_CLIENTS = 10

def load_datasets(num_clients: int):
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("../clients/", train=True, download=True, transform=transform)
    testset = CIFAR10("../clients/", train=False, download=True, transform=transform)

    # Assign class labels to each partition based on a probability distribution
    classes = trainset.classes
    print(classes)
    class_probs = [0.1, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]  # Example distribution

    partition_classes = []
    for _ in range(num_clients):
        partition_classes.append(np.random.choice(classes, size=len(trainset), p=class_probs))

    partition_classes = [set(partition) for partition in partition_classes]

    # Split training set into `num_clients` partitions to simulate different local datasets
    datasets = []
    for i in range(num_clients):
        indices = np.where(partition_classes[i] == classes)[0]
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


# def load_data():
#     """Load CIFAR-10 (training and test set)."""
#     transform = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     )
#     trainset = CIFAR10(".", train=True, download=True, transform=transform)
#     testset = CIFAR10(".", train=False, download=True, transform=transform)
#     trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
#     testloader = DataLoader(testset, batch_size=32)
#     num_examples = {"trainset": len(trainset), "testset": len(testset)}
#     return trainloader, testloader, num_examples


class NCFClient(fl.client.NumPyClient):
    def __init__(self, cid, model, trainloader, testloader, num_examples):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_examples = num_examples

    def train(self, epochs):
        """Train the model on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        for _ in range(epochs):
            for images, labels in tqdm(self.trainloader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()

    def test(self):
        """Test the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for data in tqdm(self.testloader):
                images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return loss, accuracy

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        self.set_parameters(parameters)
        self.train(epochs=1)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        self.set_parameters(parameters)
        # TODO : Change to Get the Hit Ratio and NDCG
        loss, accuracy = self.test()
        print(f"Loss->{loss} Accuracy->{accuracy}")
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


# if __name__ == '__main__':
#     # Load model and data
#     net = Net().to(DEVICE)
#     trainloader, testloader, num_examples = load_data()
#
#     fl.client.start_numpy_client(server_address="localhost:8080",
#                                  client=NCFClient(model=net,
#                                                   trainloader=trainloader,
#                                                   testloader=testloader,
#                                                   num_examples=num_examples
#                                                   )
#
#                                  )

trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS)

def numpyclient_fn(cid) -> NCFClient:
    net = Net().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    num_examples = {"trainset": len(trainloader), "testset": len(valloader)}
    return NCFClient(cid=cid, model=net, trainloader=trainloader, testloader=valloader, num_examples=num_examples)