import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the movie ratings dataset
ratings_data = pd.read_csv('ratings.csv')

# Create a user-movie matrix
user_movie_matrix = ratings_data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# Calculate the similarity matrix using cosine similarity
similarity_matrix = pd.DataFrame(cosine_similarity(user_movie_matrix))

# Function to get top N movie recommendations for a given user
def get_movie_recommendations(user_id, top_n=5):
    user_ratings = user_movie_matrix.loc[user_id].values.reshape(1, -1)
    similarity_scores = cosine_similarity(user_ratings, user_movie_matrix)[0]
    top_movie_indices = similarity_scores.argsort()[::-1][:top_n]
    top_movie_ids = user_movie_matrix.columns[top_movie_indices]
    movie_recommendations = ratings_data[ratings_data['movie_id'].isin(top_movie_ids)]['movie_title'].unique()
    return movie_recommendations

# Example usage
user_id = 1
recommendations = get_movie_recommendations(user_id, top_n=5)
print(f"Recommended movies for user {user_id}:")
for movie in recommendations:
    print(movie)
