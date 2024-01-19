# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Author(s):  Enrico Stefanel (enrico.stefanel@studenti.units.it)
Date:       2024-01-18
"""


import pickle
import logging
from typing import List, Tuple, Union
import numpy as np
from tabulate import tabulate
from utils.matfac import WeightedMatrixFactorization
from sklearn.metrics.pairwise import cosine_similarity


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RecommenderSystem():

    def __init__(self, name:str="", reviews=None, users=list(), items=list()) -> None:
        self.name = name
        self.users = users
        self.items = items
        self.reviews = reviews

        self.user_embedding = None
        self.item_embedding = None

        if reviews is not None:
            self.build_embeddings()


    def build_embeddings(self) -> None:
        logger.debug("Building embeddings...")

        wmf = WeightedMatrixFactorization(ratings=self.reviews)
        self.user_embedding, self.item_embedding = wmf.fit()


    def save(self, filename:str=None) -> None:
    
        logger.debug(f"Saving {self.name} to filesystem...")

        if not filename:
            logger.debug("No filename provided, using default (object_name.pkl)...")
            filename = f"{self.name}.pkl"

        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename:str=None) -> object:
            
            logger.debug(f"Loading {self.name} from filesystem...")
    
            if not filename:
                logger.debug("No filename provided, using default (object_name.pkl)...")
                filename = f"{self.name}.pkl"
    
            with open(filename, 'rb') as f:
                return pickle.load(f)


    def print_user_chart(self, user:str=None, first_n:int=10) -> None:

        user_chart = self.get_user_chart(user, first_n)
        
        print(tabulate(zip(range(1,first_n+1), user_chart[:,0], user_chart[:,1]), headers=['Position', 'Item Name', 'Rating'], tablefmt='rst'))

        return

    def get_user_chart(self, user:str=None, first_n:int=10):
        # Retunr the list of items that the user has reviewed,
        # sorted by the rating

        if user not in self.users:
            raise ValueError(f"User {user} not found!")

        # Get the user index
        user_index = self.users.index(user)

        # Get the user's ratings
        user_ratings = self.reviews[user_index]

        # Get the indices of the items that the user has reviewed
        reviewed_item_indices = np.where(user_ratings > 0)[0]

        user_reviews = [[self.items[i], user_ratings[i]] for i in reviewed_item_indices]

        return np.array(sorted(user_reviews, key=lambda x: x[1], reverse=True))[:first_n]



    def get_user_recommendations(self, user:str="", recommendation_num:int=10, contentbased_perc:float=.5):
        """
        Recommend top-k items for a user based on user and item embeddings.

        Input(s):   user:               The user for which we want to get the recommendations
                    recommendation_num: The number of recommendations to return
                    contentbased_perc:  The percentage of content-based suggestions to return

        Output(s):  recommended_items: List of top-k recommended item indices

        Example:    get_user_recommendations(user="User 100", top_k=10, contentbased_perc=.6)
        """

        # Check if the provided user exists
        if user not in self.users:
            raise ValueError(f"User {user} not found!")

        # Check if the provided percentage is valid
        if contentbased_perc < 0 or contentbased_perc > 1:
            raise ValueError(f"The percentage of content-based suggestions must be between 0% and 100% (got {contentbased_perc*100}%).")
        
        # Check if the embeddings have been built. Otherwise, suggest to run build_embeddings()
        if self.user_embedding.size == 0 or self.item_embedding.size == 0:
            raise ValueError("User and item embeddings not found! Please run build_embeddings() first.")
        
        contentbased_items = int(contentbased_perc * recommendation_num)
        collaborative_items = recommendation_num - contentbased_items

        # Get the user index
        user_index = self.users.index(user)

        contentbased_items = self.contentbased_filtering(user_index, contentbased_items)
        collaborative_item = self.collaborative_filtering(user_index, collaborative_items)

        print(contentbased_items)

        return None


    def contentbased_filtering(self, user:str="", rec_len:int=10, print_chart:bool=False) -> Union[List[Tuple[int, float]], None]:
        """
        Recommend top-k items for a user based on user and item embeddings.

        Input(s):   user:           The user for which we want to get the recommendations
                    rec_len:        The number of recommendations to return
                    print_chart:    Whether to print the recommendations in a table or not
                                    (if not, return a list of tuples (item, similarity))

        Ouput(s):   recommended_items: List of top-k recommended item indices if print_chart=False,
                                    None if print_chart=True
        """

        # Check if the provided user exists
        if user not in self.users:
            raise ValueError(f"User {user} not found!")
        
        # Check if the embeddings have been built. Otherwise, suggest to run build_embeddings()
        if self.user_embedding.size == 0 or self.item_embedding.size == 0:
            raise ValueError("User and item embeddings not found! Please run build_embeddings() first.")

        # Get the user index
        user_idx = self.users.index(user)

        target_user_embedding = self.user_embedding[user_idx]
        item_indices = np.where(np.isnan(self.reviews[user_idx]))[0]

        # Calculate cosine similarity between the target user and items based on content features
        similarities = cosine_similarity([target_user_embedding], self.item_embedding[item_indices]).flatten()

        # Sort items and similarities together according to content-based similarity in descending order
        sorted_indices = np.argsort(similarities)[::-1][:rec_len]
        recommended_item_indices = item_indices[sorted_indices]
        recommended_items_similarities = similarities[sorted_indices]

        # Return a list of tuples (item index, similarity)
        recommended_items = list(zip(recommended_item_indices, recommended_items_similarities))

        if print_chart:
            # Print the table with position, item name and similarity

            # Unzip the items and similarities
            recommended_items = list(zip(*recommended_items))

            # Get the item names
            recommended_items[0] = [self.items[i] for i in recommended_items[0]]

            # Get the similarities as percentages
            recommended_items[1] = [f"{sim*100:.2f}%" for sim in recommended_items[1]]

            print(f"Top {rec_len} content-based recommendations for user {user}:")
            print(tabulate(zip(range(1,rec_len+1), recommended_items[0], recommended_items[1]), headers=['Position', 'Item Name', 'Similarity'], tablefmt='rst'))
            return None
        
        return recommended_items
        

    def collaborative_filtering(self, user_index, top_k=10) -> (np.ndarray, np.ndarray):
        """
        Recommend top-k items for a user based on user-based collaborative filtering.

        Parameters:
        - user_index: Index of the target user
        - user_factors: Matrix of user embeddings
        - item_factors: Matrix of item embeddings
        - top_k: Number of items to recommend

        Returns:
        - recommended_items: List of top-k recommended item indices
        """

        # Get the embedding for the target user
        target_user_embedding = self.user_embedding[user_index]

        # Calculate cosine similarity between the target user and all other users
        similarities = cosine_similarity([target_user_embedding], self.user_embedding).flatten()

        # Get the indices of the top-k users with highest similarity (excluding the target user)
        similar_users = np.argsort(similarities)[::-1][1:top_k+1]

        # Aggregate the preferences of similar users to make recommendations
        aggregated_preferences = np.sum(self.user_embedding[similar_users] * similarities[similar_users, np.newaxis], axis=0)

        # Exclude items the user has already interacted with
        item_indices = np.where(np.isnan(self.reviews[user_index]))[0]

        # Get the indices of the top-k unrated items with highest aggregated preferences
        recommended_items = item_indices[np.argsort(aggregated_preferences)[::-1][:top_k]]

        # Get the similarities of the top-k items
        recommended_items_similarities = similarities[recommended_items]

        return zip(recommended_items, recommended_items_similarities)