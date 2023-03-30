from datetime import datetime

import flwr as fl
import torch
from flwr.server.strategy import Strategy
from torch.utils.tensorboard import SummaryWriter

from src.core.clients.client import NCFClient
from src.core.clients.dataLoader_test import load_datasets
from src.core.model.testing_model import Net
from src.utils import utils
from src.utils.vizualization import plot_metric_from_history

SERVER_WRITER = SummaryWriter(log_dir=f"runs/{datetime.now():%Y-%m-%d_%H_%M}/Server")
config = utils.get_config()


# Define metric aggregation function
# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     return {}
# server_round = metrics[0][1]["server_round"]
# for label in ["test"]:
#     # Multiply accuracy of each client by number of examples used
#     accuracies = [m[f"num_examples_{label}"] * m[f"accuracy_{label}"] for _, m in metrics]
#     losses = [m[f"num_examples_{label}"] * m[f"loss_{label}"] for _, m in metrics]
#     examples = [m[f"num_examples_{label}"] for _, m in metrics]
#     # Aggregate and return custom metric (weighted average)
#     round_accuracy = sum(accuracies) / sum(examples)
#     # round_loss = sum(losses) / sum(examples)
#     # SERVER_WRITER.add_scalar(f'Loss2/{label}', round_loss, server_round)
#     # SERVER_WRITER.add_scalar(f'Accuracy/{label}', round_accuracy, server_round)
# return {"accuracy": round_accuracy}


def numpyclient_fn(cid) -> NCFClient:
    net = Net().to(DEVICE)
    print("CID", cid)
    trainloader, valloader = trainloaders[int(cid)], valloaders[int(cid)]
    num_examples = {"trainset": len(trainloader), "testset": len(valloader)}
    return NCFClient(cid=cid,
                     model=net,
                     trainloader=trainloader,
                     testloader=valloader,
                     num_examples=num_examples,
                     )


if __name__ == '__main__':
    utils.seed_everything(int(config["Common"]["seed"]))

    DEVICE = torch.device("cpu")
    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 1}

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        # evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=lambda curr_round: {"server_round": curr_round,
                                             "local_epochs": int(config["Client"]['num_epochs'])
                                             },
        on_evaluate_config_fn=lambda curr_round: {"server_round": curr_round},
        fraction_fit=float(config["Server"]["fraction_fit"]),
        fraction_evaluate=float(config["Server"]["fraction_evaluate"]),
    )
    # # Start Flower server
    # fl.server.start_server(
    #     server_address="localhost:8080",
    #     config=fl.server.ServerConfig(num_rounds=3),
    #     strategy=strategy,
    # )

    # Create datasets
    trainloaders, valloaders, testloader = load_datasets(int(config["Common"]["num_clients"]))

    history = fl.simulation.start_simulation(
        client_fn=numpyclient_fn,
        num_clients=int(config["Common"]["num_clients"]),
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=int(config["Server"]["num_rounds"])),
        client_resources=client_resources,
    )

    plot_metric_from_history(
        history,
        SERVER_WRITER,
    )
