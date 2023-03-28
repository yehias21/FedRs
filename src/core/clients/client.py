from collections import OrderedDict
from typing import List

import flwr as fl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import numpy as np

# from src.core.servers.server import trainloaders, valloaders


DEVICE = torch.device("cpu")


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

    def set_parameters(self, parameters: List[np.ndarray]):
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

