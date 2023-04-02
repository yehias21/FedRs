import argparse
import os.path
from datetime import datetime
from typing import List

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 num_examples,
                 device,
                 log: bool = False
                 ):
        seed_everything(int(config["Common"]["seed"]))
        self.cid = cid
        # print(f'Creating Client {self.cid} ..')
        self.device = device
        self.log = log
        if self.log:
            self.writer = SummaryWriter(log_dir=f"runs/{datetime.now():%Y%m%d_%H%M}/Client{self.cid}")
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.batch_size = 32
        self.num_examples = num_examples
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=float(config["Client"]["learning_rate"]))
        self.load_client_state()

    def train(self, epochs, server_round):
        """Train the model on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        n_total_steps = len(self.train_loader)
        running_loss = 0.0
        updated_items = [0] * int(config['ml_1m']['total_items'])
        for epoch in range(epochs):
            pbar = self.train_loader
            for i, (x, y) in enumerate(pbar):
                x, y = x.to(DEVICE), y.to(DEVICE)
                # updated_items[x] = 1
                self.optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                # pbar.set_description(f"Round[{server_round}], Client[{self.cid}], Epoch [{epoch + 1}/{epochs}]")
                if (i + 1) % 100 == 0:
                    if self.log:
                        self.writer.add_scalar("running_loss",
                                               running_loss / 100,
                                               (server_round - 1) * (epoch + 1) * n_total_steps + i)
        return running_loss / epochs, updated_items

    def get_parameters(self, config):
        # print(f"[Client {self.cid}] get_parameters")
        return self.model.get_parameters()

    def set_parameters(self, parameters: List[np.ndarray]):
        # print(f"[Client {self.cid}] set_parameters")
        self.model.set_parameters(parameters)

    def fit(self, parameters, config):
        # print(f"[Client {self.cid}] fit, config: {config}")
        self.set_parameters(parameters)
        loss, updated_items = self.train(epochs=config['local_epochs'],
                                         server_round=config['server_round'])
        self.save_client_state()
        return self.get_parameters(config={}), self.num_examples["trainset"], {'loss': loss,
                                                                               'updated_items': updated_items,
                                                                               }

    def evaluate(self, parameters, config):
        # print(f"[Client {self.cid}] evaluate, config: {config}")
        self.set_parameters(parameters)
        metrics = calc_metrics(model=self.model,
                               test_loader=self.test_loader,
                               device=self.device)
        return 0.0, self.num_examples["testset"], metrics

    def save_client_state(self):
        save_path = os.path.join("./checkpoints", "clients")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        wanted_to_be_saved_params = ['embedding_user_mlp.weight', 'embedding_user_mf.weight']
        state_dict = {param_name: self.model.state_dict()[param_name] for param_name in wanted_to_be_saved_params}
        torch.save({
            'model_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
        },
            f=os.path.join(save_path, f"{self.cid}.pt"),
        )

    def load_client_state(self):
        checkpoint_path = os.path.join("./checkpoints", "clients", f"{self.cid}.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            model_dict = checkpoint['model_dict']
            for param_name in model_dict:
                if param_name in self.model.state_dict():
                    self.model.state_dict()[param_name].copy_(model_dict[param_name])


def client_fn(cid) -> NCFClient:
    net = NeuMF(config).to(DEVICE)
    loader = NCFloader(config, int(cid) + 1)
    train_loader, test_loader = loader.get_train_instance(), loader.get_test_instance()
    num_examples = {"trainset": len(train_loader) * (int(config['dataloader']['neg_samples']) + 1),
                    "testset": 100}
    return NCFClient(cid=int(cid), model=net, train_loader=train_loader, test_loader=test_loader,
                     num_examples=num_examples, device=DEVICE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cid', type=int, required=True)
    parser.add_argument('--log', type=bool, required=False, default=False)
    args = parser.parse_args()
    fl.client.start_numpy_client(server_address="localhost:8080", client=client_fn(args.cid))
