# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Author(s):  Enrico Stefanel (enrico.stefanel@studenti.units.it)
Date:       2024-01-18
"""

import numpy as np
from tqdm import tqdm
import logging


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def cosine_similarity(X, Y):
    """Returns the cosine similarity between X and Y.
    """

    try:
        return (X @ Y) / (np.linalg.norm(X) * np.linalg.norm(Y))
    except RuntimeWarning:
        return 0