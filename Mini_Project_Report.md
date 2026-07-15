# Mini-Project Report
**Title**: Movie Recommendation System using Python 

**Academic Context**: Third Year Artificial Intelligence and Data Science (SPPU Syllabus).

---

## 1. Objective
The primary objective of this project is to design and implement a content-based recommendation system that suggests movies similar to a user’s query. Furthermore, it aims to enhance user experience by retrieving live movie posters exclusively via the TMDb API and presenting the recommendations in a web application.

## 2. Problem Statement
With thousands of titles available on streaming platforms, users often experience information overload when attempting to select a movie. The problem is to develop a machine learning application that analyzes textual data (genres, keywords, cast, crew, overview) of movies a user already likes, computes similarity metrics, and accurately recommends five highly related movies to alleviate search fatigue.

## 3. Dataset Description
- **Source**: `movie_dataset.csv` (approximately 4800 movies).
- **Features Extracted**: `genres`, `keywords`, `cast`, `director`, `overview`.
- **Nature**: The dataset is relatively noisy, containing missing (`NaN`) values and unstructured text that must be aggregated, cleaned, and vectorized before mathematical comparison can occur.

## 4. Methodology

### 4.1. Data Preprocessing
The dataset is loaded into memory using the Python library Pandas. A subset of critical unstructured text features (`genres`, `keywords`, `cast`, `director`, `overview`) is selected. Missing values (nulls) within these columns are imputed with empty strings to prevent concatenation errors. Finally, the text from all five columns is concatenated row-wise into a single string (`combined_features`).

### 4.2. Feature Extraction (TF-IDF)
The concatenated text string for each movie cannot be mathematically compared in its raw state. To convert text into a measurable numerical format, Term Frequency-Inverse Document Frequency (TF-IDF) from `scikit-learn` is employed. 
`TfidfVectorizer` transforms the strings into a matrix of token counts, penalizing commonly used English stop words (like "the", "and", "is") which offer little discriminatory value, while rewarding unique descriptors.

### 4.3. Similarity Model (Cosine Similarity)
Once the data is vectorized into a matrix, **Cosine Similarity** is utilized to calculate the similarity between every pair of movies. Cosine similarity measures the cosine of the angle between two non-zero vectors. A score of `1.0` implies identical text profiles, whereas `0.0` implies complete orthogonal independence. The system queries this pre-computed similarity matrix to identify the top 5 nearest neighbors for any given movie.

### 4.4. TMDb API Integration
The recommendation engine algorithms are strictly offline, localized computations to adhere to architectural requirements. However, generating an engaging User Interface requires visual assets. When the top 5 similar movies are resolved, their titles are dynamically sent to the TMDb Search API (`api.themoviedb.org/3/search/movie`) to fetch the corresponding `poster_path`. The app renders these URLs to the frontend.

## 5. Tools & Technologies Used
- **Language**: Python 3.x
- **Data Analytics / ML**: Pandas, NumPy, Scikit-learn (TfidfVectorizer, cosine_similarity)
- **Web Framework**: Flask, Jinja2 Templates
- **Frontend Assets**: HTML5, custom dark-mode CSS, Bootstrap 5 framework for responsive grid layouts.
- **External Integration**: Requests library (for HTTP calls), python-dotenv (environment variables), TMDb API.

## 6. Results and Conclusion
The application successfully starts a local web server displaying a clean, autocomplete-enabled search interface. Upon entering a query (e.g., *Inception*), the Python backend effectively calculates the vectors, extracts the top 5 highest cosine similarity distances, independently queries the TMDb API for those precise posters, and returns the customized `result.html` template. 
The system operates rapidly in real-time, demonstrating practical implementation of natural language text processing and matrix operations encapsulated within an interactive web architecture.

## 7. Future Scope
While current operations rely solely on static content-metadata (Content-Based Filtering), the deployment could be expanded through:
- **Collaborative Filtering**: Incorporating real user ratings down the line to suggest highly rated movies enjoyed by similar demographics.
- **Performance Optimization**: Transitioning the cosine similarity matrix into a serialized binary format (Pickle) to eliminate the 2-4 second matrix computation delay upon the application's initial launch.
- **Advanced NLP**: Integrating more sophisticated embeddings like Word2Vec, BERT, or transformers to grasp semantic context better than the rigid word-matching of TF-IDF.
