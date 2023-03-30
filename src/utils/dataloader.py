import os
import shutil
import pandas as pd

def _reindex(self, ratings):
    """
    Process dataset to reindex userID and itemID, also set rating as binary feedback
    """
    user_list = list(ratings['user_id'].drop_duplicates())
    user2id = {w: i for i, w in enumerate(user_list)}

    item_list = list(ratings['item_id'].drop_duplicates())
    item2id = {w: i for i, w in enumerate(item_list)}

    ratings['user_id'] = ratings['user_id'].apply(lambda x: user2id[x])
    ratings['item_id'] = ratings['item_id'].apply(lambda x: item2id[x])
    ratings['rating'] = ratings['rating'].apply(lambda x: float(x > 0))
    return ratings

def _leave_one_out(self, ratings):
    """
    leave-one-out evaluation protocol in paper https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf
    """
    ratings['rank_latest'] = ratings.groupby(['user_id'])['timestamp'].rank(method='first', ascending=False)
    test = ratings.loc[ratings['rank_latest'] == 1]
    train = ratings.loc[ratings['rank_latest'] > 1]
    assert train['user_id'].nunique()==test['user_id'].nunique(), 'Not Match Train User with Test User'
    return train[['user_id', 'item_id', 'rating']], test[['user_id', 'item_id', 'rating']]

def _negative_sampling(self, ratings):
    interact_status = (
        ratings.groupby('user_id')['item_id']
        .apply(set)
        .reset_index()
        .rename(columns={'item_id': 'interacted_items'}))
    interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
    interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, self.num_ng_test))
    return interact_status[['user_id', 'negative_items', 'negative_samples']]
path='C:\Users\yahia_shaaban\projects\graduation projects\FedRs\data\raw\ml-1m\ml-1m\ratings.dat'
fed_path='C:\Users\yahia_shaaban\projects\graduation projects\FedRs\data\federated'
num_of_clients=1000

def dfFilter(df):
  df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
  last_interaction = df.loc[df.groupby('user')['timestamp'].idxmax()]['item'].unique()
  corr_presented = df.groupby('item')['rating'].sum()>20
  df.loc[max_timestamps['item'].unique()]
  return df

def federate_data(dataset_name):
    if os.path.isdir(os.path.join(fed_path,dataset_name)):
        files = os.listdir(os.path.join(fed_path,dataset_name))
        if len([file.endswith(".data") for file in files]) == num_of_clients :
            return
        shutil.rmtree(os.path.join(fed_path,dataset_name))
    os.makedirs(os.path.join(fed_path,dataset_name), exist_ok=True)
    # read dataset using pandas
    df = pd.read_csv("/content/ratings.dat",sep="::",names=['user','item','rating', 'timestamp'])
    df.loc[:,'rating'] = 1
    # # filter based on interactions
    filtered=dfFilter(df)
    # reindex
    reindexed=reindex(filtered)
    # # write to federated dir
    for group in reindexed.groupby("user"):
        user_id, user_data = group
        file_path = os.path.join('/content', f"user_{user_id}.csv")
        user_data = user_data.drop(['user', 'rating'], axis=1)
        user_data.to_csv(file_path, index=False)

'''
class dataloader:
def __init__(self,client_id):
    read file from fedrated
    if not available rasise error
    initalize_data;
    self.positive= indexes
    self.negtive= set(i for range(n))-set(postives)
def 

class Rating_Datset(torch.utils.data.Dataset):
	def __init__(self, user_list, item_list, rating_list):
		super(Rating_Datset, self).__init__()
		self.item_list = item_list
		self.rating_list = rating_list

	def __len__(self):
		return len(self.user_list)

	def __getitem__(self, idx):
		user = self.user_list[idx]
		item = self.item_list[idx]
		rating = self.rating_list[idx]
		
		return (
			torch.tensor(item, dtype=torch.long),
			torch.tensor(rating, dtype=torch.bool)
			)
'''
if __name__ == "__main__":
    # federate_data(data_path,output_path)