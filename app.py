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
    background: #0e1117;
    color: white;
}

h1 {
    font-size: 40px;
}

.poster {
    background: #1c1f26;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
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
st.title("🎬 Movie Recommender System Using NLP & ML")
st.write("Type or select a movie from the dropdown")

# ---------------- SELECT MOVIE (LABEL FIXED) ----------------
movie_selected = st.selectbox(
    label="Select Movie",
    options=movies["title"].values
)

# ---------------- RECOMMEND BUTTON ----------------
if st.button("🎥 Show Recommendation"):
    idx = movies[movies["title"] == movie_selected].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    st.subheader("Recommended Movies")

    cols = st.columns(5)
    for col, (i, _) in zip(cols, scores):
        with col:
            st.markdown("<div class='poster'>", unsafe_allow_html=True)
            st.image(movies.iloc[i]["poster_url"], use_container_width=True)
            st.caption(movies.iloc[i]["title"])
            st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<center>© 2025 | Movie Recommendation System | Built with ❤️ using Streamlit</center>
""", unsafe_allow_html=True)
