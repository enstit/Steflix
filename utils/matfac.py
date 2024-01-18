# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Author(s):  Enrico Stefanel (enrico.stefanel@studenti.units.it)
Date:       2024-01-18
"""


import logging
import numpy as np
from numpy.linalg import solve


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WeightedMatrixFactorization:
    def __init__(self, ratings, weight_observed=1.0, weight_unobserved=0.1, num_factors=100, lambda_reg=0, num_iterations=10):
        """
        Initialize the weighted matrix factorization model.

        Parameters:
        - ratings: User-item interaction matrix (numpy array) with NaN values.
        - weight_observed: Weight parameter for observed values.
        - weight_unobserved: Weight parameter for non-observed values.
        - num_factors: Number of latent factors for users and items.
        - lambda_reg: Regularization parameter.
        - num_iterations: Number of iterations for ALS.
        """
        self.ratings = np.nan_to_num(np.array(ratings),0)
        self.observed_data = ~np.isnan(ratings)
        self.num_users, self.num_items = self.ratings.shape
        self.weight_observed = weight_observed
        self.weight_unobserved = weight_unobserved
        self.num_factors = num_factors
        self.lambda_reg = lambda_reg
        self.num_iterations = num_iterations

        # Initialize user and item matrices with random values
        self.user_matrix = np.random.rand(self.num_users, self.num_factors)
        self.item_matrix = np.random.rand(self.num_items, self.num_factors)

    def fit(self):
        """
        Train the weighted matrix factorization model using ALS.
        """
        for iteration in range(self.num_iterations):
            self.update_user_matrix()
            self.update_item_matrix()

            # Calculate the loss (the difference between the observed ratings and the dot product of the user and item vectors)
            loss = np.sum(
                np.where(
                    self.observed_data,
                    (self.ratings - self.user_matrix @ self.item_matrix.T) ** 2,
                    0
                )
            )
            logger.debug(f"Iteration: {iteration + 1} -> Loss: {loss}")

        return self.user_matrix, self.item_matrix

    def update_user_matrix(self):
        """
        Update the user matrix using ALS.
        """
        for user in range(self.num_users):

            # Weight matrix for observed and unobserved values
            weight_matrix = np.diag(
                np.where(
                    self.observed_data[user, :],
                    self.weight_observed,
                    self.weight_unobserved
                )
            )

            # Regularization term
            regularization = self.lambda_reg * np.eye(self.num_factors)
            
            self.user_matrix[user,:] = solve(
                self.item_matrix.T @ weight_matrix @ self.item_matrix + regularization,
                self.item_matrix.T @ weight_matrix @ self.ratings[user, :]
            )
            
            # Solve the system of linear equations using spsolve
            #self.user_matrix[user, :] = spsolve(self.item_matrix.T @ weight_matrix @ self.item_matrix + regularization,
            #                                    self.item_matrix.T @ weight_matrix @ self.ratings[user, :])

    def update_item_matrix(self):
        """
        Update the item matrix using ALS.
        """
        for item in range(self.num_items):

            # Weight matrix for observed and unobserved values
            weight_matrix = np.diag(
                np.where(
                    self.observed_data[:,item],
                    self.weight_observed,
                    self.weight_unobserved
                )
            )

            # Regularization term
            regularization = self.lambda_reg * np.eye(self.num_factors)

            # Solve the system of linear equations using spsolve
            self.item_matrix[item, :] = solve(self.user_matrix.T @ weight_matrix @ self.user_matrix + regularization,
                                                self.user_matrix.T @ weight_matrix @ self.ratings[:, item])

    def predict(self, user, item):
        """
        Predict the rating for a user-item pair.

        Parameters:
        - user: User index.
        - item: Item index.

        Returns:
        - Predicted rating.
        """
        return np.dot(self.user_matrix[user, :], self.item_matrix[item, :].T)