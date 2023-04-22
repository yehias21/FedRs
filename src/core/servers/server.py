import argparse
from datetime import datetime

import flwr as fl
import torch
from torch.utils.tensorboard import SummaryWriter

from src.core.clients.client import client_fn
from src.core.servers.serverFedWAvg import SaveFedAvgStrategy
from src.utils import utils

SERVER_WRITER = SummaryWriter(log_dir=f"runs/{datetime.now():%Y%m%d_%H%M}/Server")
config = utils.get_config()

# TODO: Checkpointing on item embeddings and model parameters

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
