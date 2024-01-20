# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Recommender System class to interact with a recommender system that
uses Weighted Matrix Factorization to build user and item embeddings,
and provides content-based and collaborative filtering recommendations.

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

    def __init__(self, name:str="", reviews=None, users=None, items=None) -> None:
        self.name = name
        self.users = users if users is not None else []
        self.users_num = len(self.users)
        self.items = items if items is not None else []
        self.items_num = len(self.items)
        self.reviews = reviews if reviews is not None else np.array([])

        # The reviews matrix must be a numpy array of shape (users_num, items_num)
        assert self.reviews.shape == (self.users_num, self.items_num), f"Reviews matrix shape must be ({self.users_num}, {self.items_num})!"

        # Initialize users and items embeddings as empty numpy arrays
        self.users_embedding = np.array([])
        self.items_embedding = np.array([])


    def build_embeddings(self, **kwargs) -> None:
        """
        Perform matrix factorization to build class users and items embeddings.

        Input(s):   **kwargs:   The user for which we want to get the recommendations

        Output(s):  None
        """

        if self.reviews is None:
            raise ValueError("No reviews found! Please provide a reviews matrix.")

        logger.debug("Building embeddings...")

        wmf = WeightedMatrixFactorization(ratings=self.reviews, **kwargs)
        self.users_embedding, self.items_embedding = wmf.fit()


    def save(self, filename:str="") -> None:
        """
        Save the RecommenderSystem object to the filesystem.

        Input(s):   filename:   The name of the file to save the object to.
                                Default: {self.name}.pkl
        Output(s):  None
        """
    
        logger.debug(f"Saving {self.name} to filesystem...")

        # If no filename is provided, use the default one
        if not filename:
            logger.debug(f"No filename provided, using default {self.name}.pkl...")
            filename = f"{self.name}.pkl"

        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


    def load(self, filename:str="") -> object:
        """
        Load the RecommenderSystem object from the filesystem.

        Input(s):   filename:   The name of the file to load the object from.
                                Default: {self.name}.pkl
        Output(s):  RecommenderSystem object
        """
        
        logger.debug(f"Loading {self.name} from filesystem...")

        # If no filename is provided, use the default one
        if not filename:
            logger.debug(f"No filename provided, using default {self.name}.pkl...")
            filename = f"{self.name}.pkl"

        with open(filename, 'rb') as f:
            return pickle.load(f)


    def get_user_chart(self, user:str="", chart_len:int=10, print_chart:bool=False) -> np.array:
        """
        Get the list of items that the user has reviewed, sorted by the rating.
        If print_chart=True, print the chart in a table.

        Input(s):   user:           The user for which we want to get the chart
                    chart_len:      The number of items to return
                    print_chart:    Whether to print the chart in a table or not.
                                    Default: False

        Output(s):  user_chart:     List of top-chart_len items reviewed by the user, sorted by rating
        """
        # Return the list of items that the user has reviewed,
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

        user_chart = np.array(sorted(user_reviews, key=lambda x: x[1], reverse=True))[:chart_len]

        if print_chart:

            print(tabulate(zip(range(1,chart_len+1), user_chart[:,0], user_chart[:,1]), headers=['Position', 'Item Name', 'Rating'], tablefmt='rst'))
            return

        return user_chart


    def contentbased_filtering(self, user:str="", rec_len:int=10, print_chart:bool=False) -> Union[List[Tuple[int, float]], None]:
        """
        Recommend rec_len items for a user based on user and item embeddings.

        Input(s):   user:           The user for which we want to get the recommendations
                    rec_len:        The number of recommendations to return
                    print_chart:    Whether to print the recommendations in a table or not
                                    (if not, return a list of tuples (item_idx, similarity))

        Ouput(s):   recommended_items: List of top-k recommended item indices if print_chart=False,
                                    None if print_chart=True
        """

        # Check if the provided user exists
        if user not in self.users:
            raise ValueError(f"User {user} not found!")
        
        # Check if the embeddings have been built. Otherwise, suggest to run build_embeddings()
        if self.users_embedding.size == 0 or self.items_embedding.size == 0:
            raise ValueError("User and item embeddings not found! Please run build_embeddings() first.")

        # Get the user index
        user_idx = self.users.index(user)

        # Get the user embedding
        target_user_embedding = self.users_embedding[user_idx]

        # Get the indices of the items that the user has not reviewed
        item_indices = np.where(np.isnan(self.reviews[user_idx]))[0]

        # Calculate cosine similarity between the target user and items based on content features
        similarities = cosine_similarity([target_user_embedding], self.items_embedding[item_indices]).flatten()

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
            recommended_movies = [self.items[i] for i in recommended_item_indices]

            # Get the similarities as percentages
            expected_similarity = [f"{sim*100:.2f}%" for sim in recommended_items_similarities]

            print(f"Top {rec_len} content-based recommendations for user {user}:")
            print(tabulate(zip(range(1,rec_len+1), recommended_movies, expected_similarity), headers=['Position', 'Item Name', 'Similarity'], tablefmt='rst'))
            return None
        
        # Return a list of tuples (item index, similarity)
        return recommended_items
        

    def collaborative_filtering(self, user:str="", rec_len:int=10, print_chart:bool=False) -> Union[List[Tuple[int, float]], None]:
        """
        Recommend rec_len items for a user, based on the ratings of similar users.

        Input(s):   user:           The user for which we want to get the recommendations
                    rec_len:        The number of recommendations to return
                    print_chart:    Whether to print the recommendations in a table or not
                                    (if not, return a list of tuples (item_idx, similarity))

        Ouput(s):   recommended_items: List of rec_len recommended item indices if print_chart=False,
                                    None if print_chart=True
        """

        # Percentage of similar users (w.r.t. the total number of users)
        # to consider for collaborative filtering
        SIMILAR_USERS_PERC = 0.10 # 10%s

        # Check if the provided user exists
        if user not in self.users:
            raise ValueError(f"User {user} not found!")
        
        # Check if the embeddings have been built. Otherwise, suggest to run build_embeddings()
        if self.users_embedding.size == 0 or self.items_embedding.size == 0:
            raise ValueError("User and item embeddings not found! Please run build_embeddings() first.")

        # Get the user index
        user_idx = self.users.index(user)

        # Get the user embedding
        target_user_embedding = self.users_embedding[user_idx]

        # Get the indices of the items that the user has not reviewed
        item_indices = np.where(np.isnan(self.reviews[user_idx]))[0]

        # Calculate cosine similarity between the target user and other users
        similarities = cosine_similarity([target_user_embedding], self.users_embedding).flatten()

        # Sort users according to collaborative filtering similarity in descending order,
        # and get the top 5% users
        sorted_indices = np.argsort(similarities)[::-1][:int(SIMILAR_USERS_PERC*len(self.users))]
        recommended_user_indices = sorted_indices[1:]  # Exclude the target user

        # Aggregate preferences of similar users
        aggregated_preferences = np.sum(self.users_embedding[recommended_user_indices], axis=0)
        
        # Calculate cosine similarity between the target user group and items based on content features
        similarities = cosine_similarity([aggregated_preferences], self.items_embedding[item_indices]).flatten()

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
            recommended_movies = [self.items[i] for i in recommended_items[0]]

            # Get the similarities as percentages
            expected_similarity = [f"{sim*100:.2f}%" for sim in recommended_items[1]]

            print(f"Top {rec_len} collaborative recommendations for user {user}:")
            print(tabulate(zip(range(1,rec_len+1), recommended_movies, expected_similarity), headers=['Position', 'Item Name', 'Similarity'], tablefmt='rst'))
            return None
        
        # Return a list of tuples (item index, similarity)
        return recommended_items