import streamlit as st
from main import df, recommend

st.set_page_config(
    page_title="Movie Recommendation System 🎬",
    layout="centered"
)

st.title("🎬 Movie Recommendation System")
st.write("Content-Based Movie Recommendation using TF-IDF & Cosine Similarity")

selected_movie = st.selectbox(
    "Select a movie",
    df["title"].values
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies")
    for movie in recommendations:
        st.write("👉", movie)
