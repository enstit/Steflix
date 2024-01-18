# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Author(s):  Enrico Stefanel (enrico.stefanel@studenti.units.it)
Date:       2024-01-18
"""

import numpy as np
from tqdm import tqdm
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def cosine_similarity(A, B):
    """
    Compute the cosine similarity between two vectors.
    Input(s):   - A: First vector.
                - B: Second vector.
    Output(s):  - Cosine similarity between A and B.
    """

    logger.debug(f"Computing cosine similarity between {A} and {B}.")

    # Assert that A and B are numpy arrays of the same shape and that they are not empty
    assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray), f"A and B must be numpy arrays. A: {A}, B: {B}"
    assert A.shape == B.shape, f"A and B must have the same shape. A: {A.shape}, B: {B.shape}"

    # If the norm of A or the norm of B is 0, return 0
    if np.linalg.norm(A) * np.linalg.norm(B) == 0:
        logger.warning(f"Norm of {A} or {B} is 0. Returning cosine similarity of 0.")
        return 0

    # Compute cosine similarity between X and Y using the formula: A * B / (||A|| * ||B||)
    cos_sim = (A @ B) / (np.linalg.norm(A) * np.linalg.norm(B))

    # Assert that cosine similarity is between -1 and 1 (included)
    assert -1 <= cos_sim <= 1, f"Cosine similarity is not between -1 and 1. Cosine similarity: {cos_sim}"

    # Return cosine similarity
    return cos_sim