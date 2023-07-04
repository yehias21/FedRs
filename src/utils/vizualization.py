from flwr.server import History
from torch.utils.tensorboard import SummaryWriter

from src.utils.utils import config


def plot_metric_from_history(
        hist: History,
        tensorboard_writer: SummaryWriter,
) -> None:
    """Function to plot from Flower server History.
    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    tensorboard_writer: SummaryWriter
        Tensorboard writer to plot.
    """
    for r, loss in hist.losses_distributed:
        tensorboard_writer.add_scalar(f'Loss/Test', loss, r)
    # check if the metrics are present then plot.
    if hist.metrics_distributed:
        for metric in hist.metrics_distributed:
            for r, value in hist.metrics_distributed[metric]:
                tensorboard_writer.add_scalar(f'{metric}/Test', value, r)


class ServerWriter:
    _instance = SummaryWriter(comment="_".join([f'_Server', "ml1m", "MF-DualPEdit",
                                                f'C{config["Common"]["num_clients"]}',
                                                f'LE{config["Client"]["num_epochs"]}',
                                                f'LR{config["Client"]["learning_rate"]}',
                                                f"S{config['Common']['seed']}",
                                                ])
                              )

    def __new__(cls, *args, **kwargs) -> SummaryWriter:
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance
