
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load & Preprocess Dataset  

def load_and_clean(path="movie_dataset.csv"):
    """Load movie dataset and perform data cleaning."""
    df = pd.read_csv(path)

    # Select relevant features
    features = ["title", "genres", "keywords", "cast", "director",
                "overview", "vote_average", "vote_count",
                "popularity", "release_date", "runtime"]
    df = df[features].copy()

    
    for col in ["genres", "keywords", "cast", "director", "overview"]:
        df[col] = df[col].fillna("")

    df.dropna(subset=["title"], inplace=True)

    # Fill numeric fields with sensible defaults
    df["vote_average"] = df["vote_average"].fillna(0.0)
    df["vote_count"]   = df["vote_count"].fillna(0).astype(int)
    df["popularity"]   = df["popularity"].fillna(0.0)
    df["runtime"]      = df["runtime"].fillna(0).astype(int)

    # Extract year from release_date
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["year"] = df["year"].fillna(0).astype(int)

    # Reset index cleanly
    df.reset_index(drop=True, inplace=True)
    return df



def build_tags(df):
    """
    Combine genres, keywords, cast, director, overview into a single
    'tag' string per movie for TF-IDF vectorization.
    """
    def combine(row):
        parts = [
            row["genres"],
            row["genres"],
            row["director"],
            row["director"],
            row["keywords"],
            row["cast"],
            row["overview"]
        ]
        return " ".join(str(p).lower() for p in parts)

    df["tags"] = df.apply(combine, axis=1)
    return df


# TF-IDF Vectorization + Cosine Similarity


class MovieRecommender:


    def __init__(self, csv_path="movie_dataset.csv"):
        print("[Recommender] Loading and preprocessing dataset...")
        self.df = load_and_clean(csv_path)
        self.df = build_tags(self.df)

        print("[Recommender] Building TF-IDF matrix...")
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2)
        )
        tfidf_matrix = self.vectorizer.fit_transform(self.df["tags"])

        print("[Recommender] Computing cosine similarity matrix...")
        self.similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

        
        self.title_index = pd.Series(
            self.df.index,
            index=self.df["title"].str.lower()
        )
        print(f"[Recommender] Ready. {len(self.df)} movies loaded.")

    # public api

    def get_recommendations(self, title: str, n: int = 10):
        
        title_lower = title.strip().lower()

        # Fuzzy match if exact not found
        if title_lower not in self.title_index:
            matches = self.title_index.index[
                self.title_index.index.str.contains(title_lower, na=False)
            ]
            if len(matches) == 0:
                return None, None, f"No movie found matching '{title}'"
            title_lower = matches[0]

        idx = self.title_index[title_lower]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
            
        matched_title = self.df.iloc[idx]["title"]

        # Similarity scores for this movie against all others
        sim_scores = list(enumerate(self.similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Skip index 0 (the movie itself)
        sim_scores = sim_scores[1: n + 1]

        results = []
        for movie_idx, score in sim_scores:
            row = self.df.iloc[movie_idx]
            results.append({
                "title":         row["title"],
                "genres":        row["genres"],
                "director":      row["director"],
                "cast":          row["cast"],
                "overview":      row["overview"],
                "vote_average":  round(float(row["vote_average"]), 1),
                "vote_count":    int(row["vote_count"]),
                "year":          int(row["year"]),
                "runtime":       int(row["runtime"]),
                "similarity":    round(float(score) * 100, 1),
            })
        return matched_title, results, None

    def get_featured(self, n=12):
        """Return top movies by a weighted popularity+rating score."""
        df = self.df.copy()
        # Weighted score = 60% vote_average + 40% normalized popularity
        df["score"] = (
            0.6 * df["vote_average"] +
            0.4 * (df["popularity"] / df["popularity"].max() * 10)
        )
        top = df.nlargest(n, "score")
        return [
            {
                "title":        row["title"],
                "genres":       row["genres"],
                "director":     row["director"],
                "vote_average": round(float(row["vote_average"]), 1),
                "year":         int(row["year"]),
            }
            for _, row in top.iterrows()
        ]

    def all_titles(self):
        """Return all movie titles (for autocomplete)."""
        return sorted(self.df["title"].dropna().tolist())

    def get_movie(self, title: str):
        """Return full info for a single movie."""
        title_lower = title.strip().lower()
        # Fallback to exact search or partial match if needed
        if title_lower not in self.title_index:
             matches = self.title_index.index[
                 self.title_index.index.str.contains(title_lower, na=False)
             ]
             if len(matches) == 0:
                 return None
             title_lower = matches[0]
             
        idx = self.title_index[title_lower]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
        row = self.df.iloc[idx]
        return {
            "title":        row["title"],
            "genres":       row["genres"],
            "director":     row["director"],
            "cast":         row["cast"],
            "overview":     row["overview"],
            "vote_average": round(float(row["vote_average"]), 1),
            "vote_count":   int(row["vote_count"]),
            "year":         int(row["year"]),
            "runtime":      int(row["runtime"]),
            "popularity":   round(float(row["popularity"]), 2),
        }
