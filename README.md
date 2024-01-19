<div align="center">
  <img src="./assets/steflix.png" width="225px">
  <h1 align="center">🍿 STEFLIX: A Movie-based Recommender System</h2>
</div>

Given a dataset containing items and users (what items the user has liked), we use Weighted Matrix Factorisation to build a Recommender System that can suggest new documents to a user using both *Content-based filtering* and *Collaborative filtering*.

The system accepts as an input *a user* (i.e., a set of liked documents), and returns a ranking of documents the user might like.

## Usage

The [RecommenderSystem Jupyter notebook](./RecommenderSystem.ipynb) contains all the information to build a custom Recommender System, and provides an example of Movies RS (data is taken from the files inside the [data](./data/) folder).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.