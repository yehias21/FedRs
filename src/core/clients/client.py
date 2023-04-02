import argparse
from datetime import datetime
from typing import List

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.core.model.model import NeuMF
from src.utils.evaluate import calc_metrics
from src.utils.mldataset import NCFloader
from src.utils.utils import get_config, seed_everything

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = get_config()


class NCFClient(fl.client.NumPyClient):
    def __init__(self,
                 cid: int,
                 model: NeuMF,
                 trainloader: DataLoader,
                 testloader: DataLoader,
                 num_examples,
                 device,
                 log: bool = False
                 ):
        seed_everything(int(config["Common"]["seed"]))
        self.device = device
        self.cid = cid
        print(f'Creating Client {self.cid} ..')
        self.log = log
        if self.log:
            self.writer = SummaryWriter(log_dir=f"runs/{datetime.now():%Y%m%d_%H%M}/Client{self.cid}")

        self.model = model
        self.train_loader = trainloader
        self.test_loader = testloader

        self.batch_size = 32
        self.num_examples = num_examples
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=float(config["Client"]["learning_rate"]))

    def train(self, epochs, server_round) -> float:
        """Train the model on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        n_total_steps = len(self.train_loader)
        running_loss = 0.0
        for epoch in range(epochs):
            pbar = tqdm(self.train_loader)
            for i, (x, y) in enumerate(pbar):
                x, y = x.to(DEVICE), y.to(DEVICE)
                self.optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                pbar.set_description(f"Round[{server_round}], Client[{self.cid}], Epoch [{epoch + 1}/{epochs}]")
                if (i + 1) % 100 == 0:
                    if self.log:
                        self.writer.add_scalar("running_loss",
                                               running_loss / 100,
                                               (server_round - 1) * (epoch + 1) * n_total_steps + i)
        return running_loss

    def get_parameters(self, config):
        # print(f"[Client {self.cid}] get_parameters")
        return self.model.get_parameters()

    def set_parameters(self, parameters: List[np.ndarray]):
        # print(f"[Client {self.cid}] set_parameters")
        self.model.set_parameters(parameters)

    def fit(self, parameters, config):
        # print(f"[Client {self.cid}] fit, config: {config}")
        self.set_parameters(parameters)
        loss = self.train(epochs=config['local_epochs'], server_round=config['server_round'])
        return self.get_parameters(config={}), self.num_examples["trainset"], {'loss': loss}

    def evaluate(self, parameters, config):
        # print(f"[Client {self.cid}] evaluate, config: {config}")
        self.set_parameters(parameters)
        metrics = calc_metrics(model=self.model,
                               test_loader=self.test_loader,
                               device=self.device)
        return 0.0, self.num_examples["testset"], metrics


def client_fn(cid) -> NCFClient:
    net = NeuMF(config).to(DEVICE)
    loader = NCFloader(config, int(cid) + 1)
    train_loader, test_loader = loader.get_train_instance(), loader.get_test_instance()
    num_examples = {"trainset": len(train_loader) * (int(config['dataloader']['neg_samples']) + 1),
                    "testset": 100}
    return NCFClient(cid=int(cid),
                     model=net,
                     trainloader=train_loader,
                     testloader=test_loader,
                     num_examples=num_examples,
                     device=DEVICE,
                     )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cid', type=int, required=True)
    parser.add_argument('--log', type=bool, required=False, default=False)
    args = parser.parse_args()
    fl.client.start_numpy_client(server_address="localhost:8080", client=client_fn(args.cid))
