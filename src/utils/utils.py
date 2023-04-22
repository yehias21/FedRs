import configparser
import os
import random
import re

import flwr as fl
import numpy as np
import torch
from flwr.common import Parameters


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_config():
    # Load the config file
    config = configparser.ConfigParser()
    config.read('./src/config.ini')
    return config


def read_latest_params(checkpoints_path: str = "./checkpoints"):
    # Find all saved weights files
    weights_files = []
    for filename in os.listdir(checkpoints_path):
        match = re.match(r"\d{8}_\d{6}", filename)
        if match:
            timestamp = match.group()
            weights_files.append((timestamp, filename))

    # Sort the files by timestamp
    weights_files.sort(reverse=True)

    # Load the last weights file, if any
    if weights_files:
        filename = weights_files[0][1]
        data = np.load(os.path.join(checkpoints_path, filename))
        parameter_arrays = [data[f"arr_{i}"] for i in range(len(data.files))]
        return fl.common.ndarrays_to_parameters(parameter_arrays)

    # No weights files found
    return None
