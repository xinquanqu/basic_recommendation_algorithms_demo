from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy
import numpy as np
import pandas as pd
import sklearn

from data.data_process import Dataset


# Content Based Model
class ContentBasedRecommender:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.item_ids = dataset.articles_df['contentId'].tolist()
        self.items_df = dataset.articles_df
        self.build_token_embeddings()
        self.user_profiles = self.build_user_profiles()

    def build_token_embeddings(self):
        #Ignoring stopwords (words with no semantics) from English and Portuguese (as we have a corpus with mixed languages)
        stopwords_list = stopwords.words('english') + stopwords.words('portuguese')
        
        #Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
        vectorizer = TfidfVectorizer(analyzer='word',
             ngram_range=(1, 2),
             min_df=0.003,
             max_df=0.5,
             max_features=5000,
             stop_words=stopwords_list)
        self.tfidf_matrix = vectorizer.fit_transform(self.dataset.articles_df['title'] + "" + self.dataset.articles_df['text'])
        
    def get_model_name(self):
        return self.MODEL_NAME


    def build_user_profiles(self):
        def get_item_profile(item_id):
            idx = self.item_ids.index(item_id)
            item_profile = self.tfidf_matrix[idx:idx+1]
            return item_profile
        
        def get_item_profiles(ids):
            item_profiles_list = [get_item_profile(x) for x in ids]
            item_profiles = scipy.sparse.vstack(item_profiles_list)
            return item_profiles
        
        def build_users_profile(person_id, interactions_indexed_df):
            interactions_person_df = interactions_indexed_df.loc[person_id]
            user_item_profiles = get_item_profiles(interactions_person_df['contentId'])
            
            user_item_strengths = np.array(interactions_person_df['eventStrength']).reshape(-1,1)
            #Weighted average of item profiles by the interactions strength
            user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
            user_item_strengths_weighted_avg = np.asarray(user_item_strengths_weighted_avg)
            user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
            return user_profile_norm
        interactions_train_df = self.dataset.interactions_train_df
        interactions_indexed_df = interactions_train_df[interactions_train_df['contentId'] \
                                                       .isin(self.dataset.articles_df['contentId'])].set_index('personId')
        user_profiles = {}
        for person_id in interactions_indexed_df.index.unique():
            user_profiles[person_id] = build_users_profile(person_id, interactions_indexed_df)
        return user_profiles
    
    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        #Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(self.user_profiles[person_id], self.tfidf_matrix)
        #Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        #Sort the similar items by similarity
        similar_items = sorted([(self.item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        #Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['contentId', 'recStrength']) \
                                    .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]


        return recommendations_df
