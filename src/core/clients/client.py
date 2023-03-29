import argparse
from collections import OrderedDict
from datetime import datetime
from typing import List

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.core.clients.dataLoader_test import load_data
from src.core.model.testing_model import Net
from src.utils.utils import get_config

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = get_config()


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
        print(f'Creating Client {self.cid} ..')
        self.log = log
        if self.log:
            self.writer = SummaryWriter(log_dir=f"runs/{datetime.now():%Y-%m-%d_%H:%M}/Client{self.cid}")

        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

        self.batch_size = 32
        self.num_examples = num_examples
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=float(config["Client"]["learning_rate"]))

    def train(self, epochs, server_round):
        """Train the model on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        n_total_steps = len(self.trainloader)
        for epoch in range(epochs):
            running_loss = 0.0
            pbar = tqdm(self.trainloader)
            for i, (images, labels) in enumerate(pbar):
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
            test_res = dict()
            for label, dataloader in [["Test", self.testloader], ["Train", self.trainloader]]:
                for data in tqdm(dataloader, desc=f"Client[{self.cid}] Testing {label} Data .. "):
                    images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                    outputs = self.model(images)
                    loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accuracy = correct / total
                test_res[label] = {"accuracy": accuracy,
                                   "loss": loss}
        return test_res

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        print(f"[Client {self.cid}] set_parameters")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        self.set_parameters(parameters)
        self.train(epochs=config['local_epochs'], server_round=config['server_round'])
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        self.set_parameters(parameters)
        # TODO : Change to Get the Hit Ratio and NDCG
        test_res = self.test()

        loss_test = test_res["Test"]["loss"]
        accuracy_test = test_res["Test"]["accuracy"]

        loss_train = test_res["Train"]["loss"]
        accuracy_train = test_res["Train"]["accuracy"]

        return float(loss_test), self.num_examples["testset"], {"accuracy_test": float(accuracy_test),
                                                                "loss_test": float(loss_test),
                                                                "num_examples_test": self.num_examples["testset"],
                                                                "accuracy_train": float(accuracy_train),
                                                                "loss_train": float(loss_train),
                                                                "num_examples_train": self.num_examples["trainset"],
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
