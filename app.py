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

# ---------------- TMDB API KEY ----------------
# Local run → paste key directly
# Streamlit Cloud → use secrets.toml
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except:
    TMDB_API_KEY = "PASTE_YOUR_TMDB_API_KEY_HERE"

# ---------------- THEME ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #3e2723, #1b0f0a);
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

/* FORCE LABEL VISIBILITY */
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] p {
    color: white !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Button */
.stButton > button {
    background: #6d4c41;
    color: white;
    padding: 12px 24px;
    border-radius: 8px;
    font-size: 16px;
    border: none;
}
.stButton > button:hover {
    background: #8d6e63;
}

/* Movie title */
.movie-title {
    text-align: center;
    font-weight: 600;
    margin-top: 6px;
}

.footer {
    text-align: center;
    margin-top: 60px;
    opacity: 0.8;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SAFE TMDB FETCH ----------------
@st.cache_data(show_spinner=False)
def fetch_movies():
    url = (
        "https://api.themoviedb.org/3/movie/popular"
        f"?api_key={TMDB_API_KEY}&language=en-US&page=1"
    )

    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            st.error("❌ TMDB API error. Check your API key.")
            return pd.DataFrame()

        data = response.json()

        if "results" not in data:
            st.error("❌ No movie data received from TMDB.")
            return pd.DataFrame()

        movies = []
        for m in data["results"]:
            movies.append({
                "title": m["title"],
                "overview": m.get("overview", ""),
                "poster": (
                    "https://image.tmdb.org/t/p/w500" + m["poster_path"]
                    if m.get("poster_path") else None
                ),
                "genres": m.get("genre_ids", [])
            })

        return pd.DataFrame(movies)

    except requests.exceptions.RequestException:
        st.error("🌐 Network error. Please check internet or try again later.")
        return pd.DataFrame()

# ---------------- LOAD MOVIES ----------------
movies = fetch_movies()

if movies.empty:
    st.stop()

# ---------------- HYBRID RECOMMENDER ----------------
tfidf = TfidfVectorizer(stop_words="english")
overview_matrix = tfidf.fit_transform(movies["overview"])
overview_similarity = cosine_similarity(overview_matrix)

def genre_similarity(g1, g2):
    return len(set(g1) & set(g2)) / max(len(set(g1) | set(g2)), 1)

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>🎬 Netflix-Style Movie Recommender</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Hybrid Recommendation using TMDB API, NLP & Genres</div>", unsafe_allow_html=True)

# ---------------- MOVIE SELECT ----------------
selected_movie = st.selectbox(
    "Select a movie",
    movies["title"].values
)

# ---------------- RECOMMEND ----------------
if st.button("🍿 Recommend Movies"):
    idx = movies[movies["title"] == selected_movie].index[0]

    # Show selected movie
    st.subheader("🎬 Selected Movie")
    if movies.iloc[idx]["poster"]:
        st.image(movies.iloc[idx]["poster"], width=250)
    st.caption(selected_movie)

    # Calculate hybrid scores
    scores = []
    for i in range(len(movies)):
        if i != idx:
            score = (
                0.7 * overview_similarity[idx][i] +
                0.3 * genre_similarity(movies.iloc[idx]["genres"], movies.iloc[i]["genres"])
            )
            scores.append((i, score))

    top_movies = sorted(scores, key=lambda x: x[1], reverse=True)[:5]

    st.subheader("✨ Recommended Movies")
    cols = st.columns(5)

    for col, (i, _) in zip(cols, top_movies):
        with col:
            if movies.iloc[i]["poster"]:
                st.image(movies.iloc[i]["poster"], width=180)
            st.markdown(
                f"<div class='movie-title'>{movies.iloc[i]['title']}</div>",
                unsafe_allow_html=True
            )

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
© 2025 • Netflix-Style Movie Recommendation System  
Built with ❤️ by Tejaswini
</div>
""", unsafe_allow_html=True)
