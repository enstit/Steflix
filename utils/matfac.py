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
    """
    Weighted Matrix Factorization model. It uses Alternating Least Squares (ALS) to train the model.
    """

    def __init__(self, ratings, weight_observed:float=1.0, weight_unobserved:float=1.0, num_factors:int=100, lambda_reg:float=0.95, num_iterations:int=20):
        """
        Initialize the weighted matrix factorization model.

        Input(s):   ratings:               The ratings matrix. It must be a numpy array.
                    weight_observed:       Weight for observed ratings. Default: 1.0
                    weight_unobserved:     Weight for unobserved ratings. Default: 0.1
                    num_factors:           Number of factors. Default: 100
                    lambda_reg:            Regularization term. Default: 0.95
                    num_iterations:        Number of iterations. Default: 10

        Output(s):  None
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


    def fit(self, method:str="ALS", **kwargs):
        """
        Train the weighted matrix factorization model using one of the
        implemented methods.

        Input(s):   - method: Method to use for training. Default: ALS (Alternating Least Squares)
                    - **kwargs: Keyword arguments for the training method.

        Output(s):  - None
        """

        if method == "ALS":
            self.__fit_als(**kwargs)
            return self.user_matrix, self.item_matrix
        else:
            raise NotImplementedError(f"Method {method} not supported. Please choose one of the following: ['ALS']")


    def __fit_als(self):

        for iteration in range(self.num_iterations):
            self.__update_user_matrix()
            self.__update_item_matrix()

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

    def __update_user_matrix(self):
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
            

    def __update_item_matrix(self):
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