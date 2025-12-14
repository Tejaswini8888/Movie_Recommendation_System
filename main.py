# Movie Recommendation System (Content-Based Filtering)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the dataset
df = pd.read_csv("movies.csv")

# 2. Combine relevant text columns (genres + overview)
df["combined_text"] = df["genres"].fillna("") + " " + df["overview"].fillna("")

# 3. Convert text to numeric vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

# 4. Compute cosine similarity between all movies
similarity_matrix = cosine_similarity(tfidf_matrix)

# 5. Recommendation Function
def recommend(movie_title, num_recommendations=5):

    # Check if movie exists
    if movie_title not in df["title"].values:
        return ["Movie not found in database"]

    # Get index of the movie
    idx = df[df["title"] == movie_title].index[0]

    # Get similarity scores
    scores = list(enumerate(similarity_matrix[idx]))

    # Sort movies by similarity score
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Get top N recommendations (excluding the movie itself)
    top_movies = sorted_scores[1:num_recommendations + 1]

    # Return movie titles
    return [df.iloc[i[0]].title for i in top_movies]
