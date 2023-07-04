from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.utils.mldataset import NCFloader
from src.utils.utils import get_config


# TODO: work on the problem of user tensors & gradient updates

class NeuMF(nn.Module):
    def __init__(self, args):
        super(NeuMF, self).__init__()
        self.num_items = int(args['ml_1m']['total_items'])
        self.latent_dim = int(args['model']['latent'])

        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.embedding_item.weight, std=0.01)

        nn.init.xavier_uniform_(self.affine_output.weight)

        # self.affine_output.bias.data.zero_()

    def forward(self, item_indices):
        # FIXME: off-by-one error in item indices in the data loader [Issue #9]
        try:
            item_indices -= 1
            item_embedding = self.embedding_item(item_indices)
            logits = self.affine_output(item_embedding)
            rating = self.logistic(logits)
            return rating.squeeze()
        except Exception as e:
            print(item_indices)
            raise e

    def get_parameters(self):
        params = []
        excluded_params = ['embedding_item.weight']
        for item, val in self.state_dict().items():
            if item in excluded_params:
                params.append(val.cpu().numpy())
        return params

    def set_parameters(self, parameters: List[np.ndarray]):
        param_names = list(self.state_dict().keys())
        excluded_params = ['embedding_item.weight']
        state_dict = OrderedDict()
        i = 0
        for key in param_names:
            if key in excluded_params:
                state_dict[key] = torch.tensor(parameters[i])
                i += 1
            else:
                state_dict[key] = self.state_dict()[key]
        self.load_state_dict(state_dict, strict=True)


if __name__ == '__main__':
    config = get_config()
    loader = NCFloader(config, 1)
    train_loader = loader.get_train_instance()
    model = NeuMF(config)
    dataiter = iter(train_loader)
    x, y = next(dataiter)
    y_dash = model(x)
    print(y_dash)
