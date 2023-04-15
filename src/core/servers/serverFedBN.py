from typing import Callable, List, Optional, Tuple, Union, Dict
import torch
from torch import nn
import torch.nn.functional as F

import flwr as fl
from flwr.server.strategy import Strategy
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
# from src.core.model.testing_model import Net, get_parameters


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class FedBN(Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

    def __repr__(self) -> str:
        return "FedBN"

    def initialize_parameters(self, client_manager) -> Dict[str, Any]:
        """Initialize global model parameters."""
        net = Net()
        return net.state_dict()

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom configs
        n_clients = len(clients)
        half_clients = n_clients // 2
        standard_config = {"lr": 0.001}
        higher_lr_config = {"lr": 0.003}
        fit_configurations = []
        for idx, client in enumerate(clients):
            if idx < half_clients:
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                fit_configurations.append((client, FitIns(parameters, higher_lr_config)))

        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average and Batch Normalization statistics."""

        # Aggregate parameters
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate batch normalization statistics
        bn_stats_list = [fit_res.extra.get("bn_stats", None) for _, fit_res in results]
        bn_stats_aggregated = self.aggregate_bn_stats(bn_stats_list)

        metrics_aggregated = {}
        return parameters_aggregated, {"bn_stats": bn_stats_aggregated, **metrics_aggregated}

    def aggregate_bn_stats(self, bn_stats_list: List[Optional[NDArrays]]) -> Optional[NDArrays]:
        """ Aggregate the running_mean and running_var of BatchNorm layers
    from multiple models.

        Args:
            bn_stats_list (List[Optional[NDArrays]]): List of batch normalization statistics from all clients.

        Returns:
            Optional[NDArrays]: Aggregated batch normalization statistics if bn_stats_list is not empty, else None.
        """
        if len(bn_stats_list) == 0 or all(stat is None for stat in bn_stats_list):
            return None

        # Initialize sum and count for mean and variance
        sum_mean = torch.zeros_like(bn_stats_list[0][0])
        sum_var = torch.zeros_like(bn_stats_list[0][1])
        count = 0

        # Aggregate mean and variance across all clients
        for mean, var in bn_stats_list:
            if mean is not None and var is not None:
                sum_mean += mean
                sum_var += var
                count += 1

        # Compute the final mean and variance
        mean = sum_mean / count
        var = sum_var / count

        # Return the final batch normalization statistics
        return mean, var

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []

        # Define the config for evaluation
        config = {"eval_batch_size": 64, "epochs": 1}

        # Create an evaluation instance
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def num_evaluation_clients(self, num_available: int) -> Tuple[int, int]:
        """Calculate the number of clients to use for evaluation."""
        if self.fraction_evaluate >= 1.0:
            num_clients = int(self.fraction_evaluate)
            return num_clients, num_clients
        elif self.fraction_evaluate <= 0.0:
            return 0, 0
        else:
            num_clients = max(int(self.fraction_evaluate * num_available), 1)
            return num_clients, 1

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""

        # Let's assume we won't perform the global model evaluation on the server side.
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

