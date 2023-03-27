from typing import List, Tuple

import flwr as fl
import torch
from flwr.common import Metrics
from src.core.clients.client import numpyclient_fn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10
import numpy as np


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)


#
# # Start Flower server
# fl.server.start_server(
#     server_address="localhost:8080",
#     config=fl.server.ServerConfig(num_rounds=3),
#     strategy=strategy,
# )




fl.simulation.start_simulation(
    client_fn=numpyclient_fn,
    num_clients=2,
    strategy= strategy,
    config=fl.server.ServerConfig(num_rounds=3),
    client_resources=client_resources,
)
