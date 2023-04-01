from datetime import datetime
from typing import List, Tuple, Union, Dict, Optional

import flwr as fl
import numpy as np
import torch
from flwr.common import Metrics, FitRes, Scalar, Parameters, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from torch.utils.tensorboard import SummaryWriter

from src.core.clients.client import NCFClient
from src.core.clients.dataLoader_test import load_datasets
from src.core.model.testing_model import Net
from src.utils import utils

SERVER_WRITER = SummaryWriter(log_dir=f"runs/{datetime.now():%Y%m%d_%H%M}/Server")
config = utils.get_config()


class SaveFedAvgStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters and server_round % 10 == 0:
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            print(f"Saving round {server_round} aggregated weights...")
            np.savez_compressed(f"./checkpoints/{datetime.now():%Y%m%d_%H%M%S}_server-round-{server_round}-weights.npz",
                                *aggregated_ndarrays)
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        SERVER_WRITER.add_scalar(f'Loss/test', aggregated_loss, server_round)

        for label in ["test"]:
            # Multiply accuracy of each client by number of examples used
            accuracies = [r.metrics[f"num_examples_{label}"] * r.metrics[f"accuracy_{label}"] for _, r in results]
            examples = [r.metrics[f"num_examples_{label}"] for _, r in results]
            # Aggregate and return custom metric (weighted average)
            round_accuracy = sum(accuracies) / sum(examples)
            SERVER_WRITER.add_scalar(f'Accuracy/{label}', round_accuracy, server_round)

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": round_accuracy}


def client_fn(cid) -> NCFClient:
    net = Net().to(DEVICE)
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
    strategy = SaveFedAvgStrategy(
        on_fit_config_fn=lambda curr_round: {"server_round": curr_round,
                                             "local_epochs": int(config["Client"]['num_epochs'])
                                             },
        on_evaluate_config_fn=lambda curr_round: {"server_round": curr_round},
        fraction_fit=float(config["Server"]["fraction_fit"]),
        fraction_evaluate=float(config["Server"]["fraction_evaluate"]),
        initial_parameters=utils.read_latest_params(),
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
        client_fn=client_fn,
        num_clients=int(config["Common"]["num_clients"]),
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=int(config["Server"]["num_rounds"])),
        client_resources=client_resources,
    )
