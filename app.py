import streamlit as st
from main import df, recommend

# Page Config
st.set_page_config(
    page_title="Movie Recommendation System 🎬",
    page_icon="🎬",
    layout="centered"
)

# ----------- BROWN THEME CSS -----------
st.markdown("""
<style>
/* App background */
.stApp {
    background-color: #3E2723;
    color: #F5F5F5;
}

/* Title */
h1, h2, h3 {
    color: #FFCCBC;
    text-align: center;
}

/* Selectbox & Button */
div[data-baseweb="select"] > div {
    background-color: #5D4037;
    color: white;
}

.stButton > button {
    background-color: #8D6E63;
    color: white;
    border-radius: 10px;
    font-size: 16px;
    padding: 8px 20px;
}

.stButton > button:hover {
    background-color: #A1887F;
    color: black;
}

/* Recommendation text */
.reco {
    background-color: #4E342E;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 8px;
    text-align: center;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ----------- APP UI -----------
st.title("🎬 Movie Recommendation System")
st.write("🤎 Content-Based Filtering using TF-IDF & Cosine Similarity")

selected_movie = st.selectbox(
    "🎥 Select a Movie",
    df["title"].values
)

if st.button("🍿 Recommend Movies"):
    recommendations = recommend(selected_movie)

    st.subheader("✨ Recommended Movies")
    for movie in recommendations:
        st.markdown(f"<div class='reco'>🎞 {movie}</div>", unsafe_allow_html=True)
