from datetime import datetime
from typing import List, Tuple

import flwr as fl
import torch
from flwr.common import Metrics
from torch.utils.tensorboard import SummaryWriter

from src.core.clients.client import NCFClient
from src.core.clients.dataLoader_test import load_datasets
from src.core.model.testing_model import Net

SERVER_WRITER = SummaryWriter(log_dir=f"runs/{datetime.now():%Y-%m-%d_%H:%M}/Server")


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    server_round = metrics[0][1]["server_round"]
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    round_accuracy = sum(accuracies) / sum(examples)
    round_loss = sum(losses) / sum(examples)
    SERVER_WRITER.add_scalar('loss', round_loss, server_round)
    SERVER_WRITER.add_scalar('accuracy', round_accuracy, server_round)
    return {"accuracy": round_accuracy}


def fit_config(server_round: int):
    """Return training configuration dict for each round.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 1,
    }
    return config


def eval_config(server_round: int):
    """Return evaluation configuration dict for each round.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
    }
    return config


def numpyclient_fn(cid) -> NCFClient:
    net = Net().to(DEVICE)
    print("CID", cid)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    num_examples = {"trainset": len(trainloader), "testset": len(valloader)}
    return NCFClient(cid=cid, model=net, trainloader=trainloader, testloader=valloader, num_examples=num_examples)


if __name__ == '__main__':
    # Define strategy
    strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average,
                                         on_fit_config_fn=fit_config,
                                         on_evaluate_config_fn=eval_config,
                                         )
    # # Start Flower server
    # fl.server.start_server(
    #     server_address="localhost:8080",
    #     config=fl.server.ServerConfig(num_rounds=3),
    #     strategy=strategy,
    # )

    DEVICE = torch.device("cpu")
    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 1}

    NUM_CLIENTS = 10

    # Create datasets
    trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS)
    # print(len(trainloaders))

    fl.simulation.start_simulation(
        client_fn=numpyclient_fn,
        num_clients=5,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=2),
        client_resources=client_resources,
    )
