import os
import random
import shutil

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils.utils import get_config


class train(torch.utils.data.Dataset):
    def __init__(self, item_list, negative_list, neg_samples):
        super(train, self).__init__()
        self.item_list = list(item_list)
        self.negative_list = list(negative_list)
        self.neg_samples = neg_samples  # +1 HERE FOR pos ITEM

    def __len__(self):
        return len(self.item_list * self.neg_samples)

    def __getitem__(self, idx):
        if idx % self.neg_samples == 0:
            item = self.item_list[idx // self.neg_samples]
            rating = 1
        else:
            item = random.choice(self.negative_list)
            rating = 0
        return torch.tensor(item, dtype=torch.int), torch.tensor(rating, dtype=torch.float)


class NCFloader:
    def __init__(self, args, client_id, eval=False, loov=True) -> None:
        path = os.path.join(args['dataloader']['federated_path'], f"user_{client_id}.csv")
        data = pd.read_csv(path)
        self.positive = data['items']
        self.neg = set(i for i in range(1, int(args['ml_1m']['total_items']) + 1)) - set(self.positive)
        test_idx = data['timestamp'].idxmax()
        self.test = data.iloc[test_idx]
        self.positive.drop(test_idx, inplace=True)
        self.args = args

    def get_train_instance(self, batch_size=8, workers=1):
        train_data = train(self.positive, self.neg, int(self.args['dataloader']['neg_samples']) + 1)
        return DataLoader(train_data,
                          batch_size=batch_size * (int(self.args['dataloader']['neg_samples']) + 1),
                          shuffle=True)

    def get_test_instance(self, workers=1):
        test_list = [self.test[0]]
        test_list.extend(random.sample(list(self.neg), 99))
        test_list = torch.tensor(test_list, dtype=torch.int)
        return DataLoader(test_list,
                          batch_size=100)


def _dfFilter(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    last_interaction = df.loc[df.groupby('user')['timestamp'].idxmax()]['item'].unique()
    corr_presented = df.groupby('item')['rating'].sum() > 20
    df.loc[max_timestamps['item'].unique()]
    return df


def _reindex(df):
    pass


def federate_data(args):
    out_path = args['dataloader']['federated_path']
    data_path = args['dataloader']['raw_path']
    if os.path.isdir(out_path):
        files = os.listdir(out_path)
        if len([file.endswith(".data") for file in files]) == args['ml_1m']['total_clients']:
            return
        shutil.rmtree(out_path)
    os.makedirs(out_path, exist_ok=True)
    # read dataset using pandas
    # df = pd.read_csv(os.path.join(data_path, 'ratings.dat'), sep="::", names=['users', 'items', 'ratings', 'timestamp'])
    df = pd.read_csv(os.path.join(data_path, 'u.data'), names=['users', 'items', 'ratings', 'timestamp'])
    df.loc[:, 'ratings'] = 1

    ''' 
    This functions still not finished
    # filter based on interactions
    # filtered=_dfFilter(df)
    # reindex
    # reindexed=_reindex(filtered)
    '''
    # write to federated dir
    # for group in reindexed.groupby("users"):
    for group in df.groupby("users"):
        user_id, user_data = group
        file_path = os.path.join(out_path, f"user_{user_id}.csv")
        user_data = user_data.drop(['users', 'ratings'], axis=1)
        user_data.to_csv(file_path, index=False)


if __name__ == '__main__':
    config = get_config()
    federate_data(config)
    loader = NCFloader(config, 1)
    train_loader = loader.get_train_instance()
    test_loader = loader.get_test_instance()
    count = 0
    for batch in train_loader:
        _, labels = batch
        count += labels.sum()
    print(count)
    for batch in test_loader:
        print(batch)
