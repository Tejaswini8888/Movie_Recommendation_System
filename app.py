import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

# ---------------- BROWN THEME CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #3e2723, #1b0f0a);
    color: #ffffff;
}

/* Titles */
.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}
.subtitle {
    text-align: center;
    opacity: 0.9;
    margin-bottom: 30px;
}

/* Labels FIX */
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] p {
    color: #ffffff !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Selectbox & inputs */
.stSelectbox div,
.stNumberInput input {
    color: #2b1a0f !important;
}

/* Button */
.stButton > button {
    background: #6d4c41;
    color: white;
    font-size: 16px;
    padding: 12px 22px;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}
.stButton > button:hover {
    background: #8d6e63;
}

/* Movie cards */
.movie-card {
    text-align: center;
}
.movie-title {
    margin-top: 8px;
    font-weight: 600;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 50px;
    opacity: 0.8;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
movies = pd.read_csv("movies.csv")

# ---------------- NLP MODEL ----------------
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["overview"])
similarity = cosine_similarity(tfidf_matrix)

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>🎬 Movie Recommender System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Using Natural Language Processing & Machine Learning</div>", unsafe_allow_html=True)

# ---------------- MOVIE SELECTION ----------------
movie_selected = st.selectbox(
    "🎥 Select a movie",
    movies["title"].values
)

# ---------------- RECOMMEND BUTTON ----------------
if st.button("🍿 Show Recommendations"):

    idx = movies[movies["title"] == movie_selected].index[0]

    # Selected movie
    st.subheader("🎬 Selected Movie")
    st.image(movies.iloc[idx]["poster_url"], width=250)
    st.caption(movie_selected)

    # Recommendations
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    st.subheader("✨ Recommended Movies")

    cols = st.columns(len(scores))
    for col, (i, _) in zip(cols, scores):
        with col:
            st.image(movies.iloc[i]["poster_url"], use_container_width=True)
            st.markdown(
                f"<div class='movie-title'>{movies.iloc[i]['title']}</div>",
                unsafe_allow_html=True
            )

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
© 2025 Movie Recommender • Built with ❤️ by Tejaswini
</div>
""", unsafe_allow_html=True)
