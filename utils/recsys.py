# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Author(s):  Enrico Stefanel (enrico.stefanel@studenti.units.it)
Date:       2024-01-18
"""


import logging
import numpy as np
from tabulate import tabulate
from utils.matfac import WeightedMatrixFactorization
from sklearn.metrics.pairwise import cosine_similarity


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RecommenderSystem():

    def __init__(self, reviews=None, users=list(), items=list()) -> None:
        self.users = users
        self.items = items
        self.reviews = reviews

        self.user_embedding = None
        self.item_embedding = None
        self.build_embeddings()


    def build_embeddings(self):
        logger.debug("Building embeddings...")
        self.user_embedding, self.item_embedding = WeightedMatrixFactorization(self.reviews).fit()


    def print_user_chart(self, user:str=None, first_n:int=10):

        user_chart = self.get_user_chart(user, first_n)
        
        print(tabulate(zip(range(1,first_n+1), user_chart[:,0], user_chart[:,1]), headers=['Position', 'Movie Name', 'Rating'], tablefmt='rst'))

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


    def get_user_recommendations(self, user:str=None, top_k:int=5):

        if user not in self.users.astype(str):
            raise ValueError(f"User {user} not found!")
        
        # Get the user index
        user_index = self.users.index(user)

        return self.collaborative_filtering(user_index, top_k)
        



    def contentbased_filtering(self, user_embedding, item_embeddings, top_k=5):
        """
        Recommend top-k items for a user based on user and item embeddings.

        Parameters:
        - user_embedding: Embedding vector for the user
        - item_embeddings: Matrix of item embeddings
        - top_k: Number of items to recommend

        Returns:
        - recommended_items: List of top-k recommended item indices
        """
        # Calculate cosine similarity between the user and all items
        similarities = cosine_similarity([user_embedding], item_embeddings).flatten()

        # Get the indices of the top-k items with highest similarity
        recommended_items = np.argsort(similarities)[::-1][:top_k]

        return recommended_items


    def collaborative_filtering(self, user_index, top_k=5):
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
        target_user_embedding = self.user_embedding[user_index].T

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

        return recommended_items