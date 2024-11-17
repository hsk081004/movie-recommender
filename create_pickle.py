import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import os

# Load the movies and credits datasets
movies = pd.read_csv('movies.csv')
credits = pd.read_csv('credits.csv')

# Merge movies and credits data on the 'title' column
movies = movies.merge(credits, on='title')

# Function to extract genres/keywords/cast/crew as a list
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

# Apply the convert function to extract genres, keywords, and cast
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert)

# Limit cast to top 3 actors/actresses
movies['cast'] = movies['cast'].apply(lambda x: x[0:3])

# Function to extract the director from the crew data
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

# Apply the fetch_director function to the 'crew' column
movies['crew'] = movies['crew'].apply(fetch_director)

# Remove rows with missing data
movies.dropna(inplace=True)

# Convert lists in columns to space-separated strings
movies['genres'] = movies['genres'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
movies['keywords'] = movies['keywords'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
movies['cast'] = movies['cast'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
movies['crew'] = movies['crew'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")

# Concatenate the text columns into a single 'tags' column
movies['tags'] = movies['overview'] + " " + movies['genres'] + " " + movies['keywords'] + " " + movies['cast'] + " " + movies['crew']

# Remove rows where 'tags' is empty or only whitespace
movies = movies[movies['tags'].str.strip().notna()]

# Optionally replace empty tags with a default value (e.g., 'no_tags')
movies['tags'] = movies['tags'].apply(lambda x: x if x.strip() else 'no_tags')

# Remove rows where 'tags' only contain stop words
def contains_only_stopwords(text):
    return all(word in ENGLISH_STOP_WORDS for word in text.split())

# Filter out rows with only stop words
movies = movies[~movies['tags'].apply(contains_only_stopwords)]

# Ensure non-empty tags (if not, replace with 'no_tags')
movies['tags'] = movies['tags'].apply(lambda x: x if len(x.strip()) > 0 else 'no_tags')

# Remove non-alphanumeric characters (if necessary)
movies['tags'] = movies['tags'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

# Remove rows where 'tags' is still empty after cleaning
movies = movies[movies['tags'].str.strip().notna()]

# Check for empty tags
print("Number of empty tags left:", movies['tags'].isna().sum())

# Use CountVectorizer to convert text data into a matrix of token counts
cv = CountVectorizer(max_features=5000, stop_words='english')

# Join all words into a string properly (ensure no single characters)
movies['tags'] = movies['tags'].apply(lambda x: ' '.join(x.split()))

# Check if there are any empty tags remaining
print("First few tags:", movies['tags'].head())

# Fit and transform the tags column into a vector representation
vector = cv.fit_transform(movies['tags']).toarray()

# Check if the vectorization worked by printing the shape of the resulting matrix
print("Vector shape:", vector.shape)

# Compute cosine similarity matrix
similarity = cosine_similarity(vector)

# Create the 'model' directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the movie list and similarity matrix as pickle files
pickle.dump(movies[['movie_id', 'title', 'tags']], open('model/movie_list.pkl', 'wb'))
pickle.dump(similarity, open('model/similarity.pkl', 'wb'))

print("Pickle files created successfully!")
