# Personal Movie Recommender

A personal movie recommendation system built in Python using the MovieLens dataset and my own movie ratings.

The project explores several recommendation approaches, evaluates their performance, and selects an item-item collaborative filtering model as the final recommender.

The goal is to build a recommender that can learn from my own ratings and generate useful recommendations for movies I have not rated yet.

## Project Overview

Rather than relying only on movie genres or trying to train a personalized model from a relatively small number of ratings, the final approach uses collective behaviour from the MovieLens dataset to learn relationships between movies. My own ratings are then used to personalize those relationships.

The recommender also provides an explanation for each recommendation by showing movies I previously rated highly that contributed to the recommendation.

## Dataset

The project uses the [MovieLens Latest Small dataset](https://grouplens.org/datasets/movielens/), which contains:

* 100,836 ratings
* 9,742 movies
* 610 users
* Movie titles and genres
* User-movie rating interactions

My personal rating data currently contains 199 rated movies.

## Project Structure

The notebook follows the development and evaluation of several recommendation approaches:

| Section | Approach                          | Purpose                                            |
| ------- | --------------------------------- | -------------------------------------------------- |
| A–H     | BPR collaborative filtering       | Benchmark collaborative filtering on MovieLens     |
| I1–I7   | Personalized BPR                  | Test whether BPR can directly learn my preferences |
| J1–J3   | Genre-based recommender           | Test content-based personalization                 |
| K1–K11  | Item-item collaborative filtering | Develop and evaluate the final recommender         |

## Model Development

### 1. MovieLens BPR

The first approach uses Bayesian Personalized Ranking (BPR) on MovieLens.

BPR learns user and movie embeddings and optimizes the model so that movies a user interacted positively with receive higher scores than sampled negative movies.

A popularity-based recommender was used as a baseline.

| Model      | HR@10 | MAP@10 | NDCG@10 |
| ---------- | ----: | -----: | ------: |
| Popularity | 0.314 |  0.033 |   0.068 |
| BPR        | 0.433 |  0.037 |   0.086 |

BPR outperformed the popularity baseline across all three metrics.

### 2. Personalized BPR

The next step attempted to personalize BPR using my own movie ratings.

This approach performed poorly because there were only 83 positive personal ratings, with 66 used for training. This is a very small amount of data from which to learn a new user representation reliably.

The model was therefore retained as an experiment and baseline rather than used as the final recommender.

### 3. Genre-based Recommender

A content-based approach was then tested using movie genres.

The approach created a representation of my genre preferences and used genre similarity to identify movies that matched those preferences.

This approach was too coarse for effective personalization. Movies with the same genre combination could receive identical representations, resulting in limited differentiation between candidates.

The model therefore performed poorly on the personal holdout set and was not selected as the final approach.

### 4. Item-item Collaborative Filtering

The final approach uses item-item collaborative filtering.

Instead of learning my preferences from only my own ratings, the model first learns relationships between movies using the much larger MovieLens user-movie interaction matrix.

My own ratings are then used to determine which of these learned relationships are most relevant to me.

The process is:

```text
MovieLens ratings
       ↓
User-movie interaction matrix
       ↓
Item-item cosine similarity
       ↓
Movies similar to my rated movies
       ↓
Weight by my ratings
       ↓
Remove movies I have already rated
       ↓
Rank recommendations
```

This approach combines the scale of MovieLens with my personal preferences.

## Final Recommender

The item-item collaborative filtering model was selected as the final recommender because it performed well on the personal holdout evaluation and, more importantly, produced recommendations that were qualitatively aligned with my actual movie preferences.

The final model uses:

| Component                 | Configuration                     |
| ------------------------- | --------------------------------- |
| Recommendation method     | Item-item collaborative filtering |
| Similarity measure        | Cosine similarity                 |
| Neighbourhood size        | 100 nearest neighbours            |
| Personal rating weighting | `(rating - 3)^2`                  |
| Candidate filtering       | Exclude already-rated movies      |
| Personal ratings          | 199                               |

The model was also tested with positive-only rating weighting:

```text
max(rating - 3, 0)^2
```

This produced identical results on the personal holdout set, so the original weighting was retained.

## Evaluation

The personal model was evaluated using an 80/20 holdout of positively rated movies.

The final item-item model achieved:

| Metric  | Result |
| ------- | -----: |
| HR@10   |  1.000 |
| MAP@10  |  0.263 |
| NDCG@10 |  0.433 |

These results should be interpreted cautiously because the personal evaluation contains only 17 held-out positive ratings.

HR@10 of 1.000 means that at least one held-out positive movie appeared in the top 10 recommendations. It does not mean that all held-out movies were successfully recommended.

The recommendations were therefore also evaluated qualitatively based on whether they matched my actual movie preferences.

## Example Recommendations

Using all 199 personal ratings, the current top 10 recommendations are:

| Rank | Movie                     |
| ---: | ------------------------- |
|    1 | The Matrix (1999)         |
|    2 | The Godfather (1972)      |
|    3 | The Sixth Sense (1999)    |
|    4 | Memento (2000)            |
|    5 | Snatch (2000)             |
|    6 | Back to the Future (1985) |
|    7 | American Beauty (1999)    |
|    8 | A Beautiful Mind (2001)   |
|    9 | The Big Lebowski (1998)   |
|   10 | Donnie Darko (2001)       |

The recommendations show several recurring patterns in my ratings, including crime and thriller movies, psychological and mystery films, high-concept science fiction, and character-driven dramas.

The model also provides explanations for individual recommendations. For example:

> The Matrix was recommended because I rated Fight Club, The Lord of the Rings: The Return of the King, and Pulp Fiction highly.

These explanations are based on the movie similarities learned from MovieLens and the contribution of my highly rated movies to each recommendation.

## Limitations

The main limitation is the relatively small amount of personal rating data. With 199 ratings, the recommender cannot yet capture all aspects of my preferences reliably.

The personal evaluation is also based on a small holdout set of 17 positive ratings, so the reported metrics should not be interpreted as a definitive measure of recommendation quality.

The item-item similarities are learned from MovieLens rather than from my personal viewing history. This makes the system possible with limited personal data, but also means that the recommendations are influenced by patterns in the broader MovieLens user population.

## Future Improvements

As more personal ratings are added, the recommender can be evaluated and refined further.

Potential future improvements include:

* Incorporating additional movie metadata beyond genres
* Combining collaborative and content-based signals
* Adding rating confidence or recency
* Testing additional similarity and weighting strategies
* Building a simple interactive interface
* Tracking recommendations and subsequent ratings over time

## Technologies

* Python
* pandas
* NumPy
* scikit-learn
* PyTorch
* SciPy
* Jupyter Notebook
* MovieLens dataset

## Project Goal

This project is both a personal recommendation system and a practical machine learning portfolio project.

The focus is not only on achieving a high evaluation score, but on comparing different approaches, understanding their limitations, selecting an appropriate model for sparse personal data, and producing recommendations that are actually useful to me.
