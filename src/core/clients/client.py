import torch
from collections import OrderedDict
import flwr as fl
import numpy as np


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NCFClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, val_data, optimizer, batch_size, learning_rate):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def train(self, trainloader, epochs):
        """Train the model on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for _ in range(epochs):
            for images, labels in trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train(self.model, trainloader, epochs=1)
        return self.get_parameters(config={}), num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, testloader)
        return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}