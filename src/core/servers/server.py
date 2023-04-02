import argparse
from datetime import datetime
from typing import List, Tuple, Union, Dict, Optional

import flwr as fl
import numpy as np
import torch
from flwr.common import Metrics, FitRes, Scalar, Parameters, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from torch.utils.tensorboard import SummaryWriter

from src.core.clients.client import client_fn
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
        """Aggregate Training Loss using weighted average and save Server checkpoints every 10 rounds."""

        if not results:
            return None, {}

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # Calculating aggregated loss
        examples = [r.num_examples for _, r in results]
        losses = [r.metrics[f"loss"] * r.num_examples for _, r in results]
        round_loss = sum(losses) / sum(examples)
        SERVER_WRITER.add_scalar(f'Loss/Train', round_loss, server_round)

        # Saving the Parameters
        if results and server_round % 10 == 0:
            # print(f"Saving round {server_round} aggregated params ...")
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            # np.savez_compressed(f"./checkpoints/{datetime.now():%Y%m%d_%H%M%S}_server-round-{server_round}-weights.npz",
            #                     *aggregated_ndarrays)

        return aggregated_parameters, {"loss": round_loss}

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        examples = [r.metrics[f"num_examples"] for _, r in results]

        # Multiply accuracy of each client by number of examples used
        ndcgs = [r.metrics[f"num_examples"] * r.metrics[f"NDCG"] for _, r in results]
        round_ndcg = sum(ndcgs) / sum(examples)
        SERVER_WRITER.add_scalar(f'Evaluation/NDCG', round_ndcg, server_round)

        hrs = [r.metrics[f"num_examples"] * r.metrics[f"HR"] for _, r in results]
        round_hr = sum(hrs) / sum(examples)
        SERVER_WRITER.add_scalar(f'Evaluation/HR', round_hr, server_round)

        # Return aggregated metrics.
        return aggregated_loss, {"NDCG": round_ndcg,
                                 "HR": round_hr}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', type=int, default=1, required=False)
    args = parser.parse_args()
    utils.seed_everything(int(config["Common"]["seed"]))

    DEVICE = torch.device("cpu")
    # Define strategy
    strategy = SaveFedAvgStrategy(
        min_available_clients=int(config["Common"]["min_available_clients"]),
        on_fit_config_fn=lambda curr_round: {"server_round": curr_round,
                                             "local_epochs": int(config["Client"]['num_epochs'])},
        on_evaluate_config_fn=lambda curr_round: {"server_round": curr_round},
        initial_parameters=None,
    )

    # Start Flower server
    if args.sim:
        config_str = f'num_clients={int(config["Common"]["num_clients"])},' \
                     f'learning_rate={config["Client"]["learning_rate"]},' \
                     f'local epochs={config["Client"]["num_epochs"]}'
        SERVER_WRITER.add_text('Configuration', config_str)
        strategy.fraction_fit = 120 / int(config["Common"]["num_clients"])
        strategy.fraction_evaluate = 120 / int(config["Common"]["num_clients"])
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=int(config["Common"]["num_clients"]),
            strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=int(config["Server"]["num_rounds"])),
            client_resources={"num_gpus": 1} if DEVICE.type == "cuda" else None,
        )
    else:
        fl.server.start_server(
            server_address="localhost:8080",
            config=fl.server.ServerConfig(num_rounds=int(config["Server"]["num_rounds"])),
            strategy=strategy,
        )
