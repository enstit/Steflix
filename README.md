<div align="center">
  <img src="./assets/steflix.png" width="225px">
  <h1 align="center">🍿 STEFLIX: A Movie-based Recommender System</h2>
</div>

**Steflix** is a Recommender System that uses Weighted Matrix Factorisation to learn embeddings from movies ratings, and uses *Content-based filtering* and *Collaborative filtering* to suggest new (ranked) movies to a the user.

The Weighted Matrix Factorisation step is performed using the Alternating Least Squares algorithm[^1].


## Usage

The [RecommenderSystem Jupyter notebook](./RecommenderSystem.ipynb) contains all the information to build a custom Recommender System, and provides an example of Movies RS (data is taken from the files inside the [data](./data/) folder).


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


[^1]: https://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf