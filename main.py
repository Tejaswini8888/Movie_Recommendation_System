# Movie Recommendation System (Content-Based Filtering)


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1 Load the dataset
df = pd.read_csv("movies.csv")
print("First 5 rows:\n", df.head())

# 2 Combine relevant text columns (genres + overview)
df["combined_text"] = df["genres"] + " " + df["overview"]

# 3 Convert text to numeric vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

print("\nTF-IDF Matrix Shape:", tfidf_matrix.shape)

# 4 Compute cosine similarity between all movies
similarity_matrix = cosine_similarity(tfidf_matrix)


# 5 Recommendation Function
def recommend(movie_title, num_recommendations=10):

    # check if movie exists
    if movie_title not in df["title"].values:
        return ["Movie not found in database."]

    # get index of the movie
    idx = df[df["title"] == movie_title].index[0]

    # get similarity scores for that movie
    scores = list(enumerate(similarity_matrix[idx]))

    # sort by similarity (highest to lowest)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # get top N movies (skip first because it's the same movie)
    top_movies = sorted_scores[1:num_recommendations + 1]

    recommendations = [df.iloc[i[0]].title for i in top_movies]
    return recommendations


# 6 Test the recommender
movie_to_search = "Movie 10"
print(f"\n🎬 Recommendations for: {movie_to_search}")
print(recommend(movie_to_search))
