import pickle
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movies and similarity matrix
movies = pickle.load(open('model/movie_list.pkl', 'rb'))
similarity = pickle.load(open('model/similarity.pkl', 'rb'))

# Function to fetch movie poster using TMDB API
def fetch_poster(movie_id):
    # Replace YOUR_API_KEY with your actual TMDB API key
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=YOUR_API_KEY&language=en-US"
    try:
        response = requests.get(url)
        data = response.json()

        # Check if 'poster_path' exists and is valid
        if 'poster_path' in data and data['poster_path']:
            poster_path = data['poster_path']
            full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
            return full_path
        else:
            # Return a placeholder image if poster_path is not available
            return "https://via.placeholder.com/500x750?text=No+Image+Available"
    except Exception as e:
        # Return placeholder image if there's an error (e.g., API call fails)
        return "https://via.placeholder.com/500x750?text=Error+Fetching+Image"

# Recommendation function
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    recommended_movie_names = []
    recommended_movie_posters = []

    for i in distances[1:6]:  # Get the top 5 similar movies
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters

# Streamlit interface
st.header('Movie Recommender System')

movie_list = movies['title'].values
selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)

    # Display recommendations in a row of 5 columns
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0], use_container_width=True)
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1], use_container_width=True)
    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2], use_container_width=True)
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3], use_container_width=True)
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4], use_container_width=True)
