from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.utils.mldataset import NCFloader
from src.utils.utils import get_config


# TODO: run the model in the fed environment
# TODO: assembly
# TODO: work on the problem of user tensors & gradient updates

class NeuMF(nn.Module):
    def __init__(self, args):
        super(NeuMF, self).__init__()
        self.num_items = int(args['ml_1m']['total_items'])
        self.factor_num_mf = int(args['model']['latent'])
        self.layers = eval(args['model']['layers'])
        self.factor_num_mlp = self.layers[0] // 2
        # self.dropout = args.dropout

        self.embedding_user_mlp = nn.Embedding(num_embeddings=1, embedding_dim=self.factor_num_mlp)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num_mlp)

        self.embedding_user_mf = nn.Embedding(num_embeddings=1, embedding_dim=self.factor_num_mf)
        self.embedding_item_mf = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num_mf)

        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())

        self.affine_output = nn.Linear(in_features=self.layers[-1] + self.factor_num_mf, out_features=1)
        self.logistic = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.embedding_user_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_user_mf.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mf.weight, std=0.01)

        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        nn.init.xavier_uniform_(self.affine_output.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, item_indices):
        # FIXME: off-by-one error in item indices in the data loader [Issue #9]
        item_indices -= 1
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)
        user_idx = torch.tensor([0] * item_indices.shape[0], dtype=torch.int)
        mlp_vector = torch.cat([self.embedding_user_mlp(user_idx), item_embedding_mlp],
                               dim=-1)  # the concat latent vector
        mf_vector = torch.mul(self.embedding_user_mf(user_idx), item_embedding_mf)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating.squeeze()

    def get_parameters(self):
        params = []
        excluded_params = ['embedding_user_mlp.weight', 'embedding_user_mf.weight']
        for item, val in self.state_dict().items():
            if item in excluded_params:
                continue
            params.append(val.cpu().numpy())
        return params

    def set_parameters(self, parameters: List[np.ndarray]):
        param_names = list(self.state_dict().keys())
        excluded_params = ['embedding_user_mlp.weight', 'embedding_user_mf.weight']
        state_dict = OrderedDict()
        i = 0
        for key in param_names:
            if key not in excluded_params:
                state_dict[key] = torch.tensor(parameters[i])
                i += 1
            else:
                state_dict[key] = self.state_dict()[key]
        self.load_state_dict(state_dict, strict=True)


if __name__ == '__main__':
    config = get_config()
    # get some random training images
    loader = NCFloader(config, 1)
    train_loader = loader.get_train_instance()
    model = NeuMF(config)
    dataiter = iter(train_loader)
    x, y = next(dataiter)
    y_dash = model(x)
    print(y_dash)