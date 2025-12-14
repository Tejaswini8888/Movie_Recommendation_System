import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    page_icon="🎥",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    opacity: 0.9;
    margin-bottom: 30px;
}

.card {
    background: rgba(255,255,255,0.1);
    padding: 25px;
    border-radius: 15px;
}

.stButton > button {
    background: #ff4b4b;
    color: white;
    border-radius: 12px;
    padding: 12px 20px;
    font-size: 16px;
    border: none;
}

.stButton > button:hover {
    background: #ff6b6b;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🎬 Movie Recommendation System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Content-Based Movie Recommendations using Machine Learning</div>", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")

movies = load_data()

# ---------------- FEATURE ENGINEERING ----------------
movies['combined_features'] = movies['overview'] + " " + movies['genres']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

similarity = cosine_similarity(tfidf_matrix)

# ---------------- UI ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

movie_selected = st.selectbox(
    "🎥 Select a movie you like",
    movies['title'].values
)

recommend = st.button("🎯 Recommend Movies")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RECOMMENDATION LOGIC ----------------
def recommend_movies(movie_title):
    index = movies[movies['title'] == movie_title].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommendations = scores[1:6]
    movie_indices = [i[0] for i in recommendations]
    return movies.iloc[movie_indices][['title', 'genres']]

if recommend:
    st.subheader("🍿 Recommended Movies for You")
    results = recommend_movies(movie_selected)

    for i, row in results.iterrows():
        st.markdown(f"**🎬 {row['title']}**")
        st.caption(f"Genres: {row['genres']}")
