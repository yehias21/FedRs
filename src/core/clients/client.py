import argparse
from collections import OrderedDict
from datetime import datetime

import flwr as fl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm

from src.core.model.testing_model import Net

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


class NCFClient(fl.client.NumPyClient):
    def __init__(self,
                 cid: int,
                 model,
                 trainloader: DataLoader,
                 testloader: DataLoader,
                 num_examples,
                 log: bool = False
                 ):
        self.cid = cid
        self.writer = SummaryWriter(log_dir=f"runs/{datetime.now():%Y-%m-%d_%H:%M}/Client{self.cid}")
        print('Starting Client', self.cid)
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_examples = num_examples
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.log = log

    def train(self, epochs, server_round):
        """Train the model on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        n_total_steps = len(self.trainloader)
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(pbar := tqdm(self.trainloader)):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                self.optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                pbar.set_description(f"Round[{server_round}], Client[{self.cid}], Epoch [{epoch + 1}/{epochs}]")
                if (i + 1) % 100 == 0:
                    if self.log:
                        self.writer.add_scalar("running_loss",
                                               running_loss / 100,
                                               (server_round - 1) * (epoch + 1) * n_total_steps + i)
                    running_loss = 0.0

    def test(self):
        """Test the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for data in tqdm(self.testloader, desc=f"Client[{self.cid}] Testing .. "):
                images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return loss, accuracy

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train(epochs=config['local_epochs'], server_round=config['server_round'])
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # TODO : Change to Get the Hit Ratio and NDCG
        loss, accuracy = self.test()
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy),
                                                           "loss": float(loss),
                                                           "server_round": config["server_round"]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cid', type=int, required=True)
    parser.add_argument('--log', type=bool, required=False, default=False)
    args = parser.parse_args()
    # Load model and data
    net = Net().to(DEVICE)
    trainloader, testloader, num_examples = load_data()

    fl.client.start_numpy_client(server_address="localhost:8080",
                                 client=NCFClient(cid=args.cid,
                                                  model=net,
                                                  trainloader=trainloader,
                                                  testloader=testloader,
                                                  num_examples=num_examples,
                                                  log=args.log
                                                  )
                                 )
