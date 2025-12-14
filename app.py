import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Netflix-Style Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ---------------- API KEY ----------------
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except:
    TMDB_API_KEY = "030f032bddb6fad3b87bb2b228968ba9"

# ---------------- NETFLIX-STYLE CSS ----------------
st.markdown("""
<style>
.stApp {
    background: #141414;
    color: white;
}

.main-title {
    font-size: 44px;
    font-weight: 900;
}

.subtitle {
    opacity: 0.85;
    margin-bottom: 30px;
}

/* Labels FIX */
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] p {
    color: white !important;
    font-weight: 600 !important;
}

/* Button */
.stButton > button {
    background: #e50914;
    color: white;
    font-size: 16px;
    padding: 10px 24px;
    border-radius: 6px;
    border: none;
}
.stButton > button:hover {
    background: #f40612;
}

/* Movie card */
.movie-title {
    text-align: center;
    font-weight: 600;
    margin-top: 6px;
    font-size: 14px;
}

.footer {
    text-align: center;
    margin-top: 60px;
    opacity: 0.7;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TMDB FUNCTIONS ----------------
@st.cache_data
def fetch_movies():
    url = f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}&language=en-US&page=1"
    data = requests.get(url).json()["results"]

    movies = []
    for m in data:
        movies.append({
            "title": m["title"],
            "overview": m["overview"],
            "poster": "https://image.tmdb.org/t/p/w500" + m["poster_path"],
            "genres": m["genre_ids"]
        })
    return pd.DataFrame(movies)

movies = fetch_movies()

# ---------------- HYBRID MODEL ----------------
tfidf = TfidfVectorizer(stop_words="english")
overview_matrix = tfidf.fit_transform(movies["overview"])
overview_similarity = cosine_similarity(overview_matrix)

def genre_similarity(g1, g2):
    return len(set(g1) & set(g2)) / max(len(set(g1) | set(g2)), 1)

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>🎬 Netflix-Style Movie Recommender</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Hybrid Recommendation using TMDB API, NLP & Genres</div>", unsafe_allow_html=True)

# ---------------- SELECT MOVIE ----------------
selected_movie = st.selectbox(
    "Select a movie",
    movies["title"].values
)

# ---------------- RECOMMEND ----------------
if st.button("🍿 Recommend Movies"):
    idx = movies[movies["title"] == selected_movie].index[0]

    scores = []
    for i in range(len(movies)):
        if i != idx:
            score = (
                0.7 * overview_similarity[idx][i] +
                0.3 * genre_similarity(movies.iloc[idx]["genres"], movies.iloc[i]["genres"])
            )
            scores.append((i, score))

    top_movies = sorted(scores, key=lambda x: x[1], reverse=True)[:5]

    st.subheader("✨ Recommended for You")
    cols = st.columns(5)

    for col, (i, _) in zip(cols, top_movies):
        with col:
            st.image(movies.iloc[i]["poster"], use_container_width=True)
            st.markdown(
                f"<div class='movie-title'>{movies.iloc[i]['title']}</div>",
                unsafe_allow_html=True
            )

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
© 2025 Netflix-Style Movie Recommender | Built by Tejaswini ❤️
</div>
""", unsafe_allow_html=True)
