from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

from data.data_process import Dataset

class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, dataset: Dataset, degrees = 15):
        self.dataset = dataset
        self.NUMBER_OF_FACTORS_MF = degrees
        self.build_user_item_embeddings_with_svd()
        self.user_predicted_ratings()
        
    def get_model_name(self):
        return self.MODEL_NAME

    def build_user_item_embeddings_with_svd(self):
        #Creating a sparse pivot table with users in rows and items in columns
        users_items_pivot_matrix_df = self.dataset.interactions_train_df.pivot(index='personId', 
                                                                  columns='contentId', 
                                                                  values='eventStrength').fillna(0)
        self.users_items_pivot_matrix_df = users_items_pivot_matrix_df
        users_ids = list(users_items_pivot_matrix_df.index)
        self.users_ids = users_ids
        users_items_pivot_matrix = users_items_pivot_matrix_df.values
        users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)
        #The number of factors to factor the user-item matrix.
        #Performs matrix factorization of the original user item matrix
        U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k = self.NUMBER_OF_FACTORS_MF)

        self.user_embeddings = U
        self.item_embeddings = Vt
        self.sigma = sigma

    def user_predicted_ratings(self):
        sigma = self.sigma
        U = self.user_embeddings
        Vt = self.item_embeddings
        
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
        
        all_user_predicted_ratings
        
        all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())
        self.cf_predictions_df = pd.DataFrame(all_user_predicted_ratings_norm, columns = self.users_items_pivot_matrix_df.columns, index=self.users_ids).transpose()

        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
                                    .reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['contentId'].isin(items_to_ignore)] \
                               .sort_values('recStrength', ascending = False) \
                               .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]


        return recommendations_df