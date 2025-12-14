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

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #111 0%, #000 60%);
    color: white;
}

h1 {
    font-size: 42px;
}

.subtitle {
    font-size: 16px;
    opacity: 0.85;
    margin-bottom: 25px;
}

.stButton > button {
    background: #c62828;
    color: white;
    border-radius: 8px;
    padding: 10px 18px;
    font-size: 15px;
    border: none;
}

.stButton > button:hover {
    background: #e53935;
}

.poster {
    text-align: center;
}

.poster img {
    border-radius: 12px;
    transition: transform 0.3s;
}

.poster img:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🎬 Movie Recommender System Using NLP and ML")
st.markdown(
    "<div class='subtitle'>Type or select a movie from the dropdown</div>",
    unsafe_allow_html=True
)

# ---------------- LOAD DATA ----------------
movies = pd.read_csv("movies.csv")

# ---------------- NLP MODEL ----------------
movies["features"] = movies["overview"] + " " + movies["genres"]

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(movies["features"])
similarity = cosine_similarity(tfidf_matrix)

# ---------------- UI ----------------
movie_selected = st.selectbox(
    "",
    movies["title"].values
)

recommend_btn = st.button("Show Recommendation")

# ---------------- RECOMMENDATION LOGIC ----------------
def recommend(movie):
    idx = movies[movies["title"] == movie].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_movies = scores[1:6]
    return movies.iloc[[i[0] for i in top_movies]]

if recommend_btn:
    results = recommend(movie_selected)

    cols = st.columns(5)
    for col, (_, row) in zip(cols, results.iterrows()):
        with col:
            st.markdown("<div class='poster'>", unsafe_allow_html=True)
            st.image(row["poster"], use_container_width=True)
            st.caption(row["title"])
            st.markdown("</div>", unsafe_allow_html=True)
