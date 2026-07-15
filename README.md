# Movie Recommendation System

This mini-project is a content-based Movie Recommendation System built using Python, Flask, Scikit-learn, and the TMDb API. It aligns with the Third Year Artificial Intelligence and Data Science (SPPU) syllabus.

## Project Structure
- `app.py`: Main Flask application containing preprocessing, model building, and UI routes.
- `dataset/movie_dataset.csv`: The dataset containing ~4800 movies.
- `templates/`: HTML templates for the user interface.
- `static/`: Custom CSS files for styling.
- `requirements.txt`: Python dependencies.
- `Mini_Project_Report.md`: Full academic project report.

## Features
- **Data Preprocessing & Feature Extraction:** Uses TF-IDF vectorization on combined features (genres, keywords, cast, director, overview).
- **Cosine Similarity:** Calculates the distance between movie vectors to find the most similar items.
- **TMDb API Integration:** Fetches live movie posters based on search queries.
- **Modern UI:** Built with Bootstrap 5 and custom CSS for a beautiful dark-mode experience.

## Installation Instructions

1. **Clone or Download the Repository:**
   Ensure you have all the provided files in your working directory.

2. **Set up a Virtual Environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **TMDb API Configuration (Optional but Recommended for Posters):**
   - Head over to [The Movie Database (TMDb)](https://www.themoviedb.org/) and create an account.
   - Go to your account settings > API, and register for a developer API key.
   - Create a file named `.env` in the root of the project directory and add your key:
     ```
     
     ```
   *(Note: If no API key is provided, the application will grace­fully fallback to placeholder images).*

5. **Run the Application:**
   ```bash
   python app.py
   ```
   > The application will take a few seconds on startup to initialize the TF-IDF matrix.

6. **View the App:**
   Open a web browser and navigate to `http://127.0.0.1:5000/`.
