from datetime import datetime
from functools import reduce
from typing import List, Tuple, Union, Dict, Optional

import flwr as fl
import numpy as np
import torch
from flwr.common import FitRes, Scalar, Parameters, EvaluateRes, parameters_to_ndarrays, ndarrays_to_parameters, \
    NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

SERVER_WRITER = SummaryWriter(log_dir=f"runs/{datetime.now():%Y%m%d_%H%M}/Server")


def aggregate(results: List[Tuple[NDArrays, int]], updated_item_vectors: Tensor) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])
    total_interactions_per_item = updated_item_vectors.sum(0)
    mlp_item_embeddings_weights = torch.Tensor()
    weighted_weights = []
    for (weights, num_examples), updated_item_vector in zip(results, updated_item_vectors):
        # mlp_item_embeddings_weights = weights[0] * updated_item_vector.sum()
        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights.append([layer * num_examples for layer in weights])
    # Compute average weights of each layer
    weights_prime = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]

    return weights_prime


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

        updated_items_vectors = torch.tensor([list(r.metrics['updated_items']) for _, r in results])
        self.server.get_parameters()

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = self.mf_aggregate_fit(server_round, results, failures,
                                                                          updated_items_vectors)

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

    def mf_aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            updated_item_vectors: Tensor,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        nd_aggregated = aggregate(weights_results, updated_item_vectors)
        parameters_aggregated = ndarrays_to_parameters(nd_aggregated)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

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
