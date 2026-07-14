import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="MovieMatch", page_icon="🎬", layout="centered")
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }
     .stApp {
        background: linear-gradient(180deg, #0e1117 0%, #141821 100%);
    }
     .app-title {
        font-size: 42px;
        font-weight: 800;
        color: #e50914;
        margin-bottom: 0px;
        letter-spacing: -1px;
    }
    .app-subtitle {
        font-size: 15px;
        color: #9ca3af;
        margin-bottom: 28px;
    }

    /* Radio buttons */
    div[role="radiogroup"] label {
        background-color: #1c1f26;
        padding: 10px 18px;
        border-radius: 8px;
        margin-right: 8px;
        border: 1px solid #2a2e37;
    }

    /* Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #1c1f26 !important;
        border: 1px solid #2a2e37 !important;
        border-radius: 8px !important;
        color: white !important;
    }

    /* Number input */
    div[data-baseweb="input"] > div {
        background-color: #1c1f26 !important;
        border: 1px solid #2a2e37 !important;
        border-radius: 8px !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #e50914;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: 0.2s;
    }
    .stButton > button:hover {
        background-color: #b0060f;
        color: white;
    }

    /* Recommendation cards */
    .movie-card {
        background-color: #1c1f26;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 4px solid #e50914;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .movie-title {
        font-size: 17px;
        font-weight: 600;
        color: #ffffff;
    }
    .movie-rating {
        font-size: 13px;
        color: #9ca3af;
        background-color: #0e1117;
        padding: 4px 10px;
        border-radius: 6px;
        white-space: nowrap;
    }
    .section-label {
        color: #e5e7eb;
        font-size: 15px;
        font-weight: 600;
        margin: 18px 0 8px 0;
    }
</style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_everything():
    model = pickle.load(open('svd_model.pkl', 'rb'))
    genre_matrix = pickle.load(open('genre_matrix.pkl', 'rb'))
    movies = pd.read_csv('movies_clean.csv')
    final_sample = pd.read_csv('final_sample.csv')
    return model, genre_matrix, movies, final_sample

model, genre_matrix, movies, final_sample = load_everything()

movies = movies.reset_index(drop=True)
indices = pd.Series(movies.index, index=movies['title'])
movie_rating_counts = final_sample.groupby('movieId').size()
popular_enough = set(movie_rating_counts[movie_rating_counts >= 20].index)
active_user_ids = (
    final_sample.groupby('userId').size()
    .sort_values(ascending=False)
    .head(20)
    .index.tolist()
)
label_to_id = {f"User {chr(65+i)}": uid for i, uid in enumerate(active_user_ids)}

popular_movie_ids = movie_rating_counts.sort_values(ascending=False).head(20).index.tolist()
popular_movie_titles = movies[movies['movieId'].isin(popular_movie_ids)]['title'].tolist()


st.markdown('<div class="app-title">🎬 MovieMatch</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">A hybrid recommendation engine — collaborative filtering for known users, content-based for new ones.</div>', unsafe_allow_html=True)

mode = st.radio("", ["I'm a returning user", "I'm new here"], horizontal=True, label_visibility="collapsed")
st.write("")
if mode == "I'm a returning user":
    st.markdown('<div class="section-label">Select your profile</div>', unsafe_allow_html=True)
    friendly_name = st.selectbox("", list(label_to_id.keys()), label_visibility="collapsed")
    user_id = label_to_id[friendly_name]

    if st.button("Get My Recommendations"):
        watched = final_sample[final_sample['userId'] == user_id]['movieId'].tolist()
        all_movie_ids = movies['movieId'].unique()
        unwatched = [m for m in all_movie_ids if m not in watched and m in popular_enough]

        predictions = [(m, model.predict(user_id, m).est) for m in unwatched[:3000]]
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_movies = predictions[:5]

        st.markdown(f'<div class="section-label">Recommended for {friendly_name}</div>', unsafe_allow_html=True)
        for movie_id, pred_rating in top_movies:
            title = movies[movies['movieId'] == movie_id]['title'].values[0]
            st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">{title}</div>
                    <div class="movie-rating">{round(pred_rating, 2)} / 5.0</div>
                </div>
            """, unsafe_allow_html=True)

else:
    st.markdown('<div class="section-label">Pick a movie you already enjoy</div>', unsafe_allow_html=True)
    movie_title = st.selectbox("", popular_movie_titles, label_visibility="collapsed")

    if st.button("Get Recommendations"):
        from sklearn.metrics.pairwise import cosine_similarity

        idx = indices[movie_title]
        # Compute similarity of THIS ONE movie against all others (fast, no giant matrix needed)
        sim_scores = cosine_similarity(genre_matrix[idx], genre_matrix).flatten()
        sim_scores = list(enumerate(sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        movie_indices = [i[0] for i in sim_scores]

        st.markdown('<div class="section-label">Because you liked that, you might also enjoy</div>', unsafe_allow_html=True)
        for title in movies['title'].iloc[movie_indices]:
            st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">{title}</div>
                </div>
            """, unsafe_allow_html=True)
