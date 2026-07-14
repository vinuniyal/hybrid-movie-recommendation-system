# 🎬 Hybrid Movie Recommendation System

A movie recommendation engine that combines **Collaborative Filtering** and **Content-Based Filtering** to handle both existing users and the cold-start problem for new users — deployed as an interactive Streamlit app.

## Problem Statement

Recommendation engines are core to how platforms like Netflix and Amazon drive engagement and revenue. A common challenge is the **cold-start problem**: collaborative filtering works well for users with rating history, but fails for brand-new users or newly added movies with no data. This project builds a hybrid system that gracefully falls back to content-based recommendations when historical data is unavailable.

## Approach

The system is built in three layers:

1. **Collaborative Filtering (SVD)** — Uses matrix factorization on the user-item rating matrix to learn latent taste patterns and predict ratings for unseen movies, trained using the `scikit-surprise` library.
2. **Content-Based Filtering** — Uses TF-IDF vectorization on movie genres with cosine similarity to recommend movies similar in content, independent of rating history.
3. **Hybrid + Cold-Start Handling** — Routes users dynamically: users with sufficient rating history get collaborative recommendations; new users are served content-based recommendations from a curated shortlist of popular movies.

## Tech Stack

- **Language:** Python
- **ML/Data:** Pandas, NumPy, Scikit-learn, Scikit-surprise (SVD)
- **Frontend/Deployment:** Streamlit
- **Dataset:** MovieLens (GroupLens Research)

## Results

| Metric | Score |
|---|---|
| RMSE | 0.84 |
| MAE | 0.64 |
| Precision@5 | 82.5% |

Sparsity of the user-item matrix used: ~98%, consistent with real-world recommendation system datasets.

## How to Run Locally

```bash
git clone https://github.com/vinuniyal/hybrid-movie-recommendation-system.git
cd hybrid-movie-recommendation-system
pip install -r requirements.txt
streamlit run app.py
```

## Limitations & Future Scope

- Currently evaluated on a sampled subset of the MovieLens dataset for faster iteration; can be scaled to the full 20M-rating dataset.
- Content-based layer only uses genre metadata; incorporating movie descriptions/embeddings (e.g., via sentence-transformers) could improve similarity quality.
- Could be extended to use implicit feedback (clicks, watch time) for real-time personalization instead of only explicit ratings.

## Author

**Vinayak Uniyal**
[LinkedIn](https://www.linkedin.com/in/vinayakuniyal/) | [GitHub](https://github.com/vinuniyal)