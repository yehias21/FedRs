from logging import WARNING
from typing import List, Tuple, Union, Dict, Optional

import flwr as fl
from flwr.common import FitRes, Scalar, Parameters, EvaluateRes, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate
from torch.utils.tensorboard import SummaryWriter

from src.utils import utils
from src.utils.utils import aggregate_mf

config = utils.get_config()
SERVER_WRITER = SummaryWriter(comment=f'_C{config["Common"]["num_clients"]}_'
                                      f'LE{config["Client"]["num_epochs"]}_'
                                      f'LR{config["Client"]["learning_rate"]}')


class MF_FedAvgStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate Training Loss using weighted average and save Server checkpoints every 10 rounds."""

        if not results:
            return None, {}

        aggregated_parameters, aggregated_metrics = self.mf_aggregate_fit(server_round, results, failures)
        for metric in aggregated_metrics:
            SERVER_WRITER.add_scalar(f'{metric}/Train', aggregated_metrics[metric], server_round)

        # # Saving the Parameters
        # if results and server_round % 10 == 0:
        #     print(f"Saving round {server_round} aggregated params ...")
        #     aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
        #     np.savez_compressed(f"./checkpoints/{datetime.now():%Y%m%d_%H%M%S}_server-round-{server_round}-weights.npz",
        #                         *aggregated_ndarrays)
        return aggregated_parameters, aggregated_metrics

    def mf_aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        updated_items_results = []
        nn_weights_results = []

        # Convert and parse results
        for _, r in results:
            params = parameters_to_ndarrays(r.parameters)
            i_vectors, nn_weights = params[:2], params[2:]
            updated_items_results.append((i_vectors, list(r.metrics['updated_items'])))
            nn_weights_results.append((nn_weights, r.num_examples))

        i_aggregated = aggregate_mf(updated_items_results)
        nn_aggregated = aggregate(nn_weights_results)
        parameters_aggregated = ndarrays_to_parameters(i_aggregated + nn_aggregated)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

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
        for metric in aggregated_metrics:
            SERVER_WRITER.add_scalar(f'Evaluation/{metric}', aggregated_metrics[metric], server_round)

        # Return aggregated metrics.
        return aggregated_loss, aggregated_metrics
