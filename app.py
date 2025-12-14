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
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except:
    st.error("TMDB API key not found in secrets")
    st.stop()

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

/* BaseWeb typography override */
div[data-baseweb="typography"] {
    color: #ffffff !important;
    opacity: 1 !important;
}

/* SELECTBOX TEXT WHITE */
div[data-baseweb="select"] span {
    color: white !important;
}

/* Dropdown options */
ul[role="listbox"] li {
    color: black !important;
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

/* INTERACTIVE FOOTER BUTTONS */
.footer-btn {
    background: rgba(255, 255, 255, 0.15);
    padding: 14px 28px;
    border-radius: 14px;
    text-decoration: none;
    color: white;
    font-weight: 600;
    margin: 12px;
    display: inline-block;
    transition: all 0.3s ease;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
    cursor: pointer;
}

/* Hover animation */
.footer-btn:hover {
    transform: translateY(-6px) scale(1.05);
    background: rgba(255, 255, 255, 0.28);
    box-shadow: 0 14px 30px rgba(141, 110, 99, 0.6);
}

/* Click feedback */
.footer-btn:active {
    transform: scale(0.96);
}

::selection {
    background: #6d4c41;
    color: white;
}
::-moz-selection {
    background: #6d4c41;
    color: white;
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
            return pd.DataFrame()

        data = response.json()
        movies = []

        for m in data.get("results", []):
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

    except:
        return pd.DataFrame()

# ---------------- LOAD MOVIES ----------------
movies = fetch_movies()
if movies.empty:
    st.error("❌ TMDB API error. Check API key & clear cache.")
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

# ---------------- MOVIE SELECT (FIXED) ----------------
selected_movie = st.selectbox(
    label="",
    options=movies["title"].values,
    index=None,
    placeholder="Select a movie",
    label_visibility="collapsed"
)

# ---------------- RECOMMEND ----------------
if selected_movie and st.button("🍿 Recommend Movies"):
    idx = movies[movies["title"] == selected_movie].index[0]

    st.subheader("🎬 Selected Movie")
    if movies.iloc[idx]["poster"]:
        st.image(movies.iloc[idx]["poster"], width=250)
    st.caption(selected_movie)

    scores = []
    for i in range(len(movies)):
        if i != idx:
            score = (
                0.7 * overview_similarity[idx][i] +
                0.3 * genre_similarity(
                    movies.iloc[idx]["genres"],
                    movies.iloc[i]["genres"]
                )
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
<div style="text-align:center; margin-top:60px;">
    <a class="footer-btn" href="https://github.com/Tejaswini8888" target="_blank">👩‍💻 GitHub</a>
    <a class="footer-btn" href="https://www.linkedin.com/in/tejaswini-madarapu/" target="_blank">💼 LinkedIn</a>
    <p style="opacity:0.8; margin-top:15px;">© 2025 • Built with ❤️ by Tejaswini Madarapu</p>
</div>
""", unsafe_allow_html=True)
