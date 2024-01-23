<div align="center">
  <img src="./assets/steflix.png" width="225px">
  <h1 align="center">🍿 STEFLIX: A Movie-based Recommender System</h2>
</div>

**Steflix** is a Recommender System that uses Weighted Matrix Factorisation to learn embeddings from movies ratings, and uses *Content-based filtering* and *Collaborative filtering* to suggest new (ranked) movies to the user.

The Weighted Matrix Factorisation step (used to retrieve the *users* and *items embeddings*) is performed using the Alternating Least Squares algorithm[^1].


## Usage

The [utils](./utils/) folder contains source code used for the project:

* the [matfac.py](./utils/matfac.py) script contains the declaration of the `WeightedMatrixFactorization` class, used by the Recommender System to calculate users and items embeddings starting from the (sparse) reviews matrix of users and items;
* the [recsys.py](./utils/recsys.py) script contains the declaration of the `RecommenderSystem` class. It uses the above embeddings to perform *Content-based filtering* and *Collaborative filtering* of any user of the system.

In the [RecommenderSystem.ipynb](./RecommenderSystem.ipynb) Jupyter notebook, a `RecommenderSystem` is built starting from the reviews of 610 users to 9737 movies (data is taken from the files inside the [data](./data/) folder). From the reviews matrix, the tastes of a specific user have been analyzed as an example, and suggestions have been made that seem to reflect what is expected of a recommendation engine.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


[^1]: https://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf