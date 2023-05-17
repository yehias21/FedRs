import configparser
import os
import random
import re
from typing import List, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import Metrics, NDArrays


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


def weighted_loss(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply loss of each client by number of items used
    losses = [num_items * m["loss"] for num_items, m in metrics]
    updated_items = [num_items for num_items, _ in metrics]
    # Aggregate and return custom metric (weighted loss)
    return {"Loss": sum(losses) / sum(updated_items)}


def weighted_eval_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply evaluation metrics of each client by number of updated_items used
    updated_items = [num_items for num_items, _ in metrics]
    ndcgs = [num_items * m["NDCG"] for num_items, m in metrics]
    hrs = [num_items * m["HR"] for num_items, m in metrics]
    # Aggregate and return custom metrics (weighted NDCG and HR)
    return {"NDCG": sum(ndcgs) / sum(updated_items), "HR": sum(hrs) / sum(updated_items)}


def aggregate_mf(results: List[Tuple[NDArrays, List[int]]]) -> NDArrays:
    # TODO: handle the case where total_updated_items is zero
    aggregated = np.zeros_like(results[0][0])
    total_updated_items = np.array([up_items for _, up_items in results]).sum(axis=0).reshape(-1, 1)
    zero_indices = np.where(total_updated_items == 0)[0]

    for i, (i_vectors, up_items) in enumerate(results):
        aggregated += i_vectors * np.array(up_items).reshape(-1, 1)
    # Avoid division by zero
    if len(zero_indices) > 0:
        print(f"{len(zero_indices)} items have not been updated !!!")
        total_updated_items[zero_indices] = 1
        # Set the aggregated values to the i_vectors value for the indices where total_updated_items is zero
        for aggregated_embedding, original_embedding in zip(aggregated, results[0][0]):
            aggregated_embedding[zero_indices] = original_embedding[zero_indices]
    aggregated /= total_updated_items
    return [i for i in aggregated]


config = get_config()
