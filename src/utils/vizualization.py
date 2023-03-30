from flwr.server import History
from torch.utils.tensorboard import SummaryWriter


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
