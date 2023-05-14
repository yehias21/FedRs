from typing import Callable, Dict, Optional, Tuple

from flwr.common import MetricsAggregationFn
from flwr.common import (
    Parameters,
    Scalar,
    NDArrays,
)
from flwr.server.strategy.secagg import SecAggStrategy

from src.core.servers.serverFedWAvg import MF_FedAvgStrategy


class MF_SecAggStrategy(MF_FedAvgStrategy, SecAggStrategy):
    def __init__(
            self,
            fraction_fit: float = 0.1,
            fraction_evaluate: float = 0.1,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            sec_agg_param_dict: Dict[str, Scalar] = {},
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        MF_FedAvgStrategy.__init__(self, fraction_fit=fraction_fit,
                                   fraction_evaluate=fraction_evaluate,
                                   min_fit_clients=min_fit_clients,
                                   min_evaluate_clients=min_evaluate_clients,
                                   min_available_clients=min_available_clients,
                                   evaluate_fn=evaluate_fn,
                                   on_fit_config_fn=on_fit_config_fn,
                                   on_evaluate_config_fn=on_evaluate_config_fn,
                                   accept_failures=accept_failures,
                                   initial_parameters=initial_parameters,
                                   fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                                   evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                                   )
        self.sec_agg_param_dict = sec_agg_param_dict

    def get_sec_agg_param(self) -> Dict[str, int]:
        return self.sec_agg_param_dict.copy()
