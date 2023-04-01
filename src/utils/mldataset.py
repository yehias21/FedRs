import os
import shutil
import random
import pandas as pd
import torch

class train(torch.utils.data.Dataset): 
	def __init__(self, item_list,negative_list,neg_samples):
		super(train, self).__init__()
		self.item_list = item_list
		self.negative_list = negative_list 
		self.neg_samples=neg_samples # +1 HERE FOR pos ITEM 

	def __len__(self):
		return len(self.item_list*self.neg_samples)

	def __getitem__(self, idx):
		if idx%self.neg_samples ==  0:
			item = self.item_list[idx]
			rating = 1
		else: 
			item = random.choice(self.negative_list)
			rating = 0
		return (torch.tensor(item, dtype=torch.short),torch.tensor(rating, dtype=torch.uint8))

class NCFloader:
    def __init__(self,args,client_id,eval=False,loov = True) -> None:
        path=os.path.join(args['dataloader']['federated_path'],f"user_{client_id}.csv")
        data = pd.read_csv(path)
        self.positive = data['items']
        self.neg = set(i for i in args['ml_1m']['total_items']) - set(self.positive)
        test_idx = data ['timestamp'].idxmax()
        self.test = data.iloc[test_idx]
        self.positive.drop(test_idx, inplace = True)
        self.args=args
    def get_train_instance(self, batch_size= 30, workers= 1):
      train_data = train(self.positive,self.neg,self.args['dataloader']['neg_samples']+1)
      return torch.utils.data.DataLoader(train_data, batch_size=batch_size*(self.args['dataloader']['neg_samples']+1), shuffle=True, num_workers=workers)
    def get_test_instance(self,workers):
        test_list=[self.test[0]]
        test_list.extend(random.choice(self.neg,99))
        return torch.utils.data.DataLoader(test_list, batch_size=20, shuffle=True, num_workers=workers)


def _dfFilter(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    last_interaction = df.loc[df.groupby('user')['timestamp'].idxmax()]['item'].unique()
    corr_presented = df.groupby('item')['rating'].sum()>20
    df.loc[max_timestamps['item'].unique()]
    return df

def _reindex(df):
    pass

def _federate_data(args):
    out_path = args['dataloader']['federated_path']
    data_path = args['dataloader']['raw_path']
    if os.path.isdir(out_path):
        files = os.listdir(out_path)
        if len([file.endswith(".data") for file in files]) == args['ml_1m']['total_clients'] :
            return
        shutil.rmtree(out_path)
    os.makedirs(out_path, exist_ok=True)
    # read dataset using pandas
    df = pd.read_csv(data_path, sep="::", names=['users','items','ratings', 'timestamp'])
    df.loc[:,'ratings'] = 1

    ''' 
    This functions still not finished
    # filter based on interactions
    # filtered=_dfFilter(df)
    # reindex
    # reindexed=_reindex(filtered)
    '''
    # # write to federated dir
    # for group in reindexed.groupby("users"):
    for group in df.groupby("users"):
        user_id, user_data = group
        file_path = os.path.join(out_path, f"user_{user_id}.csv")
        user_data = user_data.drop(['users', 'ratings'], axis=1)
        user_data.to_csv(file_path, index=False)


