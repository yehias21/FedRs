from datetime import datetime
from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
from torch.utils.tensorboard import SummaryWriter

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

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
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


if __name__ == '__main__':
    # Define strategy
    strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average,
                                         on_fit_config_fn=fit_config,
                                         on_evaluate_config_fn=eval_config,
                                         )
    # Start Flower server
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
