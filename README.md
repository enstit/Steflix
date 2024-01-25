<div align="center">
  <img src="./assets/steflix.png" width="225px">
  <h1 align="center">🍿 STEFLIX: A Movie-based Recommender System</h2>
</div>

**Steflix** is a Recommender System that uses Weighted Matrix Factorisation to learn embeddings from movies ratings, and uses *Content-based filtering* and *Collaborative filtering* to suggest new (ranked) movies to the user.


## Usage

First of all, clone the repository to your local machine:

```bash
mkdir project/folder
cd project/folder
git clone git@github.com:enstit/Steflix.git
```

and then install the required Python packages in order to correctly run the code:

```bash
pip install -r requirements.txt
```

The [utils](./utils/) folder contains source code used for the project:

* the [matfac.py](./utils/matfac.py) script contains the declaration of the `WeightedMatrixFactorization` class, used by the Recommender System to calculate users and items embeddings starting from the (sparse) reviews matrix of users and items;
* the [recsys.py](./utils/recsys.py) script contains the declaration of the `RecommenderSystem` class. It uses the above embeddings to perform *Content-based filtering* and *Collaborative filtering* of any user of the system.

In the [RecommenderSystem.ipynb](./RecommenderSystem.ipynb) Jupyter notebook, a `RecommenderSystem` is built starting from the reviews of 943 users to 1682 movies (data is taken from the [MovieLens 100k](https://grouplens.org/datasets/movielens/latest/) dataset[^1]). From the reviews matrix, the tastes of a specific user have been analyzed as an example, and suggestions have been made that seem to reflect what is expected from a recommendation engine.


## Theoretical stuff

How exactly does the Recommender System work, from a theoretical point of view?

First of all, we start from a matrix $C$ that we are going to call the *reviews matrix* (or *feedbacks matrix*) of dimensions $M\times N$. Each row of the matrix correspond to a different user of the system, and each column to a particolar item (or movie, in our study case). It follows that a cell represents the rating that a particular user has given to an item (can be in a $1$/$0$ form, a rating from $1$ to $5$, $\dots$). Of course, no value means no rating left by a user to an item.

$$
C_{M\times N} =
\begin{bmatrix}
    & 3 &   & 1 & 2 &   & 5 \\
  4 &   &   & 2 &   & 2 &   \\
    & 3 & 1 &   &   & 3 & 4 \\
  2 &   & 4 &   & 3 &   & 5 \\
  5 & 2 & 2 &   & 3 &   &   \\
\end{bmatrix}
$$

Empty cells (i.e., unknown users' ratings) is exactly what we want to infer from the model. Knowing that, we would be able to suggest a user items that they would **probably** like, and keep them away from items that would **probably** dislike.

But what exactly are the *features* that a particular user considers when rating an item? If we knew them, and if we also knew how a particular item can be expressed as a combination of this *features*, we would also be able to retrieve expeced ratings for un-rated items.

It turns out that we don't need to know what this *features* are at all. Indeed, we call them *latent features*.

We can find two matrices $U$ and $V$ (respectively, the *users embedding* and the *items embedding*) with $k$ columns (with $k$ the number of *latent factors*) such that
$$
C \approx U_{M\times k} \cdot V_{N\times k}^{T}.
$$

The algorithm used for this decomposition is the Weighted Alternating Least Squares algorithm[^2], in which:
1. we start with $U$ and $V$ randomly generated;
2. we fix $U$ and fin, by solving a linear system, the best $V$;
3. we fix $V$ and fin, by solving a linear system, the best $U$;
3. we repeat 2. and 3. for a fixed number of iterations.

Basically, with this algorithm we optimise the difference between the original matrix $C$, and its approximation $C'$ by also weighting the observed and unobserved values using different values $w_{i,j}$ and $w_{0}$ to reflect the uncertainty in the projection.

Once we obtain this new matrices, we can use them to compute similarities (in our case, *cosine similarity*) of the rated items with respect to
* the unrated items, finding the ones nearest to a user tastes (*Content-based filtering*);
* the other users, finding clusters of people with similar interests (*Collaborative filtering*),from which we can extract new items to recommend to the user;
* a mix of these two methods, using an hybrid approach.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


[^1]: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. [https://doi.org/10.1145/2827872](https://doi.org/10.1145/2827872)
[^2]: https://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf