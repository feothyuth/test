import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the movie ratings dataset
ratings_data = pd.read_csv('ratings.csv')

# Load the movie metadata containing movie_id, movie_title, and genres
metadata = pd.read_csv('movies_metadata.csv', usecols=['id', 'title', 'genres'])

# Rename the 'id' column to 'movie_id' to match the ratings_data column
metadata.rename(columns={'id': 'movie_id'}, inplace=True)

# Merge the ratings_data with the metadata based on 'movie_id'
ratings_data = pd.merge(ratings_data, metadata, on='movie_id')

# Create a user-movie-genre matrix
user_movie_matrix = ratings_data.pivot_table(index='user_id', columns=['movie_id', 'title'], values='rating', fill_value=0)

# Remove the 'title' index level to get a multi-index dataframe with 'user_id' and 'movie_id' as indices
user_movie_matrix = user_movie_matrix.droplevel('title', axis=1)

# Calculate the similarity matrix using cosine similarity and genre preferences
def get_movie_recommendations(user_id, top_n=5):
    user_ratings = user_movie_matrix.loc[user_id]
    user_genres = user_ratings.dot(ratings_data['genres']).sum()  # Calculate user's genre preferences

    # Calculate the similarity matrix using both ratings and genres
    similarity_scores = cosine_similarity(user_ratings, user_movie_matrix) + user_genres

    # Find the top N similar movies
    top_movie_indices = similarity_scores.argsort()[::-1][:top_n]
    top_movie_ids = user_movie_matrix.columns[top_movie_indices]
    movie_recommendations = ratings_data[ratings_data['movie_id'].isin(top_movie_ids)]['title'].unique()
    return movie_recommendations

# Example usage
user_id = 1
recommendations = get_movie_recommendations(user_id, top_n=5)
print(f"Recommended movies for user {user_id}:")
for movie in recommendations:
    print(movie)
