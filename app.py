import os
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, url_for, flash, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_flash_messages'

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")




from recommender import MovieRecommender


print("Initializing MovieRecommender... This may take a moment.")
dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'movie_dataset.csv')
recommender = MovieRecommender(dataset_path)
all_titles = recommender.all_titles()
print("Initialization Complete.")


def fetch_poster(title):
    """
    Fetch movie poster using TMDb API.
    If no API key is set or the movie is not found, returns a placeholder.
    """
    placeholder = "https://via.placeholder.com/500x750/1c1c1e/ffffff?text=No+Poster+Available"
    
    if not TMDB_API_KEY:
        return placeholder
        
    try:
        url = f"https://api.tmdb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if data.get('results') and len(data['results']) > 0:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        print(f"Error fetching poster: {e}")
        
    return placeholder

# Routes


@app.route('/')
def home():
    return render_template('index.html', titles=all_titles)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        movie_name = request.form.get('movie_name', '').strip()
    else:
        movie_name = request.args.get('movie_name', '').strip()
    

    if not movie_name:
        flash("Please enter a movie name.", "warning")
        return redirect(url_for('home'))
        

    matched_title, recommendations, error_msg = recommender.get_recommendations(movie_name, n=10)
    
    if error_msg:
        flash(error_msg, "danger")
        return redirect(url_for('home'))
        

    recommended_movies = []
    for m in recommendations:
        title = m['title']
        poster = fetch_poster(title) 

        sim_percentage = f"{m['similarity']}%"
        
        recommended_movies.append({
            'title': title,
            'poster': poster,
            'similarity': sim_percentage,
            'overview': m['overview'][:150] + "..." 
        })
        

    input_movie_data = recommender.get_movie(matched_title)
    
    genres_list = str(input_movie_data['genres']).split() if input_movie_data and input_movie_data.get('genres') else []
    cast_str = str(input_movie_data['cast']).replace(' ', ', ') if input_movie_data and input_movie_data.get('cast') else "Unknown"
    
    input_movie = {
        'title': matched_title,
        'poster': fetch_poster(matched_title),
        'overview': input_movie_data['overview'] if input_movie_data else "",
        'vote_average': input_movie_data['vote_average'] if input_movie_data else 0.0,
        'vote_count': f"{input_movie_data['vote_count']:,}" if input_movie_data else 0,
        'year': input_movie_data['year'] if input_movie_data else "",
        'runtime': input_movie_data['runtime'] if input_movie_data else 0,
        'genres': genres_list,
        'director': input_movie_data['director'] if input_movie_data else "Unknown",
        'cast': cast_str
    }
        
    return render_template('result.html', input_movie=input_movie, recommendations=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
