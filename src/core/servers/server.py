import argparse

import flwr as fl
import torch

from src.core.clients.client import client_fn
from src.core.servers.serverFedMFSecagg import MF_SecAggStrategy
from src.utils import utils
from src.utils.utils import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', type=int, default=1, required=False)
    args = parser.parse_args()
    utils.seed_everything(int(config["Common"]["seed"]))

    DEVICE = torch.device("cpu")
    # Define strategy
    # strategy = MF_FedAvgStrategy(
    #     min_available_clients=int(config["Common"]["min_available_clients"]),
    #     on_fit_config_fn=lambda curr_round: {"server_round": curr_round,
    #                                          "local_epochs": int(config["Client"]['num_epochs'])
    #                                          },
    #     on_evaluate_config_fn=lambda curr_round: {"server_round": curr_round},
    #     fit_metrics_aggregation_fn=utils.weighted_loss,
    #     evaluate_metrics_aggregation_fn=utils.weighted_eval_metrics,
    #     # TODO: Checkpointing on item embeddings and model parameters
    #     initial_parameters=None,
    # )

    strategy = MF_SecAggStrategy(
        fraction_fit=0.9,
        fraction_evaluate=0.9,
        min_available_clients=int(config["Common"]["min_available_clients"]),
        on_fit_config_fn=lambda curr_round: {"server_round": curr_round,
                                             "local_epochs": int(config["Client"]['num_epochs'])
                                             },
        on_evaluate_config_fn=lambda curr_round: {"server_round": curr_round},
        fit_metrics_aggregation_fn=utils.weighted_loss,
        evaluate_metrics_aggregation_fn=utils.weighted_eval_metrics,
        # TODO: Checkpointing on item embeddings and model parameters
        initial_parameters=None,
    )

    # Start Flower server
    if args.sim:
        strategy.fraction_fit = 120 / int(config["Common"]["num_clients"])
        strategy.fraction_evaluate = 120 / int(config["Common"]["num_clients"])
        history = fl.simulation.start_simulation(
            client_fn=lambda cid: client_fn(int(cid) + 1),  # cid in the csv files starts from 1
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
