from typing import List, Tuple

import flwr as fl
import torch
from flwr.common import Metrics
from src.core.model.testing_model import Net
from src.core.clients.client import NCFClient
from src.core.clients.dataLoader_test import load_datasets


NUM_CLIENTS = 10

DEVICE = torch.device("cpu")

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

# Create datasets
trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS)
# print(len(trainloaders))

# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)



def numpyclient_fn(cid) -> NCFClient:
    net = Net().to(DEVICE)
    print("CID", cid)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    num_examples = {"trainset": len(trainloader), "testset": len(valloader)}
    return NCFClient(cid=cid, model=net, trainloader=trainloader, testloader=valloader, num_examples=num_examples)
#
# # Start Flower server
# fl.server.start_server(
#     server_address="localhost:8080",
#     config=fl.server.ServerConfig(num_rounds=3),
#     strategy=strategy,
# )



fl.simulation.start_simulation(
    client_fn=numpyclient_fn,
    num_clients=5,
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=2),
    client_resources=client_resources,
)
