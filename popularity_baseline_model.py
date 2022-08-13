import pandas as pd
import numpy as np
from tqdm import tqdm

val = pd.read_csv('data/val_small.csv')
test = pd.read_csv('data/test_small.csv')
train = pd.read_csv('data/train_small.csv')
movies = pd.read_csv('data/movies.csv')
# val = pd.read_csv('large/val_large.csv')
# train = pd.read_csv('large/train_large.csv')

class popularity:
    def __init__(self, ratings, beta_g=0, beta_i=0, beta_u=0):
        self.ratings = ratings
        self.beta_g = beta_g
        self.beta_u = beta_u
        self.beta_i = beta_i
        self.mapping = self.ratings.movieId.unique()
        self.utility_matrix()
        self.mu = self.utility.sum()/ (np.linalg.norm(self.utility) + self.beta_g)
    def utility_matrix(self):
        self.utility = np.zeros((self.ratings.userId.unique().shape[0], len(self.mapping)))
        for index, row in tqdm(self.ratings.iterrows()):
            self.utility[int(row.userId-1),int(np.where(self.mapping==row.movieId)[0][0])] = row.rating
    def interaction(self, user, movie):
         b_user = (self.utility[user,:].sum() - self.mu)/ (np.linalg.norm(self.utility) + self.beta_u)
        try:
            b_movie = (self.utility[:,movie].sum() - self.mu - b_user)/ (np.linalg.norm(self.utility) + self.beta_i)
            return b_movie + self.mu
        except:
            return self.mu + b_user
    def predict(self,user, pred_movie):
        res = []
        for i in tqdm(pred_movie):
            movie = np.where(self.mapping==i)[0]
            if movie.shape[0] == 0:
                movie = i
            res.append([i, self.interaction(user, i)])
        return sorted(res, key=lambda x: x[1], reverse=True)
    
    
p = popularity(train, beta_g=1, beta_i=1, beta_u=1)
pred = p.predict(0, val.movieId.unique().tolist())
df.drop(labels=1, axis=1).iloc[0:100,:].to_csv('large_test_result.csv')
df.drop(labels=1, axis=1).iloc[0:100,:].to_csv('large_val_result.csv')
