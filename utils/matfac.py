# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Weighted Matrix Factorization class to factorize a ratings matrix into two 
matrices, representing the users embeddings and the items embeddings.
The optimization is performed using WALS (Weighted Alternating Least Squares).

Author(s):  Enrico Stefanel (enrico.stefanel@studenti.units.it)
Date:       2024-01-18
"""


import logging
import numpy as np
from typing import Tuple
from numpy.linalg import solve
from tqdm import tqdm


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WeightedMatrixFactorization():
    """
    Weighted Matrix Factorization class to factorize a ratings matrix into two
    matrices, representing the users embeddings and the items embeddings.
    """

    def __init__(self, ratings, weight_observed:float=1.0, weight_unobserved:float=0.1, num_factors:int=100, lambda_reg:float=0.05, num_iterations:int=10) -> None:
        """
        Initialize the weighted matrix factorization model.

        Input(s):   ratings:               The ratings matrix. It must be a numpy array.
                    weight_observed:       Weight for observed ratings. Default: 1.0
                    weight_unobserved:     Weight for unobserved ratings. Default: 0.1
                    num_factors:           Number of factors. Default: 100
                    lambda_reg:            Regularization term. Default: 0.05
                    num_iterations:        Number of iterations. Default: 10

        Output(s):  None
        """
        self.ratings = np.nan_to_num(np.array(ratings),0) # Replace NaN values with 0
        self.observed_data = ~np.isnan(ratings) # Create a boolean matrix where True values are observed ratings, False values are unobserved ratings
        self.num_users, self.num_items = self.ratings.shape # Get the number of users and items
        self.weight_observed = weight_observed # Weight for observed ratings
        self.weight_unobserved = weight_unobserved # Weight for unobserved ratings
        self.num_factors = num_factors # Number of factors
        self.lambda_reg = lambda_reg # Regularization term
        self.num_iterations = num_iterations # Number of iterations for the fitting process

        self.is_fitted = False # Flag to check if the model has been fitted

        # Initialize user and item matrices with random values
        self.user_matrix = np.random.rand(self.num_users, self.num_factors)
        self.item_matrix = np.random.rand(self.num_items, self.num_factors)


    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the user and item embeddings.

        Input(s):   None

        Output(s):  - user_matrix: The user embeddings matrix.
                    - item_matrix: The item embeddings matrix.
        """

        if not self.is_fitted:
            raise ValueError("The ratings matrix has not been factorized yet. Please call the fit() method first.")

        return self.user_matrix, self.item_matrix


    def fit(self, method:str="WALS", **kwargs) -> None:
        """
        Train the weighted matrix factorization model using one of the
        implemented methods.

        Input(s):   - method:   Method to use for training. Default: WALS
                                (Weighted Alternating Least Squares)
                    - **kwargs: Keyword arguments for the training method.

        Output(s):  - None
        """

        # Check which method to use for the matrix factorization
        if method == "WALS":
            # Train the model using WALS (Weighted Alternating Least Squares)
            self.__fit_wals(**kwargs)
            self.is_fitted = True # Set the flag to True to indicate that the model has been fitted
            return
        else:
            # Raise an error if the method is not supported
            raise NotImplementedError(f"Method {method} not supported. Please choose one of the followings: 'WALS'.")


    def __fit_wals(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the weighted matrix factorization model using WALS (Weighted Alternating Least Squares).

        Input(s):   None

        Output(s):  None
        """

        for iteration in tqdm(range(self.num_iterations)):
            self.__update_users_matrix()
            self.__update_items_matrix()

            # Calculate the loss (the difference between the observed ratings
            # and the dot product of the user and item vectors)
            loss = np.sum(
                np.where(
                    self.observed_data,
                    (self.ratings - self.user_matrix @ self.item_matrix.T) ** 2,
                    0
                )
            )
            logger.debug(f"Iteration: {iteration + 1} -> Loss: {loss}")

        return self.user_matrix, self.item_matrix

    def __update_users_matrix(self) -> None:
        """
        Update the users matrix using WALS (Weighted Alternating Least Squares).

        Input(s):   None

        Output(s):  None
        """

        for user_idx in range(self.num_users):

            # Weight matrix for observed and unobserved values
            weight_matrix = np.diag(
                np.where(
                    self.observed_data[user_idx, :],
                    self.weight_observed / sum(self.observed_data[user_idx, :]), # Normalize the weight for observed ratings
                    self.weight_unobserved / sum(~self.observed_data[user_idx, :]) # Normalize the weight for unobserved ratings
                )
            )

            # Regularization term
            regularization = self.lambda_reg * np.eye(self.num_factors)
            
            # Solve the system of linear equations
            self.user_matrix[user_idx,:] = solve(
                self.item_matrix.T @ weight_matrix @ self.item_matrix + regularization,
                self.item_matrix.T @ weight_matrix @ self.ratings[user_idx, :]
            )
        
        return


    def __update_items_matrix(self) -> None:
        """
        Update the items matrix using WALS (Weighted Alternating Least Squares).

        Input(s):   None

        Output(s):  None
        """

        for item_idx in range(self.num_items):

            # Weight matrix for observed and unobserved values
            weight_matrix = np.diag(
                np.where(
                    self.observed_data[:,item_idx],
                    self.weight_observed / sum(self.observed_data[:, item_idx]) , # Normalize the weight for observed ratings
                    self.weight_unobserved / sum(~self.observed_data[:, item_idx]) # Normalize the weight for unobserved ratings
                )
            )

            # Regularization term
            regularization = self.lambda_reg * np.eye(self.num_factors)

            # Solve the system of linear equations using spsolve
            self.item_matrix[item_idx, :] = solve(self.user_matrix.T @ weight_matrix @ self.user_matrix + regularization,
                                                self.user_matrix.T @ weight_matrix @ self.ratings[:, item_idx])
            
        return