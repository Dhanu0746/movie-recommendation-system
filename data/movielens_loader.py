"""
MovieLens dataset loader for the recommendation system
This will load real movie data from MovieLens dataset
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import os
from typing import Dict, List, Tuple
import urllib.request

class MovieLensLoader:
    """Load and process MovieLens dataset"""
    
    def __init__(self, dataset_size='small'):
        """
        Initialize MovieLens loader
        
        Args:
            dataset_size: 'small' (100K ratings) or 'large' (25M ratings)
        """
        self.dataset_size = dataset_size
        self.data_dir = 'data/movielens'
        self.urls = {
            'small': 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
            'large': 'https://files.grouplens.org/datasets/movielens/ml-latest.zip'
        }
        
    def download_dataset(self):
        """Download MovieLens dataset"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        url = self.urls[self.dataset_size]
        zip_path = os.path.join(self.data_dir, f'ml-{self.dataset_size}.zip')
        
        if not os.path.exists(zip_path):
            print(f"Downloading MovieLens {self.dataset_size} dataset...")
            urllib.request.urlretrieve(url, zip_path)
            print("Download completed!")
        
        # Extract the dataset
        extract_path = os.path.join(self.data_dir, f'ml-{self.dataset_size}')
        if not os.path.exists(extract_path):
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print("Extraction completed!")
        
        return extract_path
    
    def load_movies(self, data_path: str) -> pd.DataFrame:
        """Load movies data"""
        movies_file = os.path.join(data_path, 'movies.csv')
        movies_df = pd.read_csv(movies_file)
        
        # Parse genres
        movies_df['genres_list'] = movies_df['genres'].str.split('|')
        
        # Extract year from title
        movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)')
        movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
        
        # Clean title (remove year)
        movies_df['title_clean'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
        
        return movies_df
    
    def load_ratings(self, data_path: str) -> pd.DataFrame:
        """Load ratings data"""
        ratings_file = os.path.join(data_path, 'ratings.csv')
        ratings_df = pd.read_csv(ratings_file)
        
        # Convert timestamp to datetime
        ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
        
        return ratings_df
    
    def load_links(self, data_path: str) -> pd.DataFrame:
        """Load links data (TMDB and IMDB IDs)"""
        links_file = os.path.join(data_path, 'links.csv')
        if os.path.exists(links_file):
            return pd.read_csv(links_file)
        return pd.DataFrame()
    
    def get_tmdb_data(self, tmdb_ids: List[int]) -> Dict:
        """Get additional movie data from TMDB API (requires API key)"""
        # This would require a TMDB API key
        # For now, return empty dict
        return {}
    
    def process_dataset(self) -> Dict:
        """Process the complete MovieLens dataset"""
        print("Loading MovieLens dataset...")
        
        # Download and extract
        data_path = self.download_dataset()
        
        # Load data
        movies_df = self.load_movies(data_path)
        ratings_df = self.load_ratings(data_path)
        links_df = self.load_links(data_path)
        
        print(f"Loaded {len(movies_df)} movies and {len(ratings_df)} ratings")
        
        # Basic statistics
        stats = {
            'total_movies': len(movies_df),
            'total_users': ratings_df['userId'].nunique(),
            'total_ratings': len(ratings_df),
            'avg_rating': ratings_df['rating'].mean(),
            'rating_distribution': ratings_df['rating'].value_counts().to_dict(),
            'year_range': (movies_df['year'].min(), movies_df['year'].max()),
            'top_genres': self._get_top_genres(movies_df)
        }
        
        return {
            'movies': movies_df,
            'ratings': ratings_df,
            'links': links_df,
            'stats': stats
        }
    
    def _get_top_genres(self, movies_df: pd.DataFrame) -> Dict:
        """Get top genres from movies"""
        all_genres = []
        for genres in movies_df['genres_list'].dropna():
            all_genres.extend(genres)
        
        genre_counts = pd.Series(all_genres).value_counts()
        return genre_counts.head(10).to_dict()
    
    def create_sample_dataset(self, n_users: int = 1000, n_movies: int = 500) -> Dict:
        """Create a smaller sample dataset for testing"""
        print("Creating sample dataset...")
        
        # Load full dataset
        full_data = self.process_dataset()
        movies_df = full_data['movies']
        ratings_df = full_data['ratings']
        
        # Sample users and movies
        sample_users = ratings_df['userId'].value_counts().head(n_users).index
        sample_movies = movies_df.sample(n=min(n_movies, len(movies_df)))
        
        # Filter ratings for sample users and movies
        sample_ratings = ratings_df[
            (ratings_df['userId'].isin(sample_users)) & 
            (ratings_df['movieId'].isin(sample_movies['movieId']))
        ]
        
        print(f"Sample dataset: {len(sample_users)} users, {len(sample_movies)} movies, {len(sample_ratings)} ratings")
        
        return {
            'movies': sample_movies,
            'ratings': sample_ratings,
            'users': sample_users,
            'stats': {
                'total_movies': len(sample_movies),
                'total_users': len(sample_users),
                'total_ratings': len(sample_ratings),
                'avg_rating': sample_ratings['rating'].mean()
            }
        }

def load_real_dataset():
    """Load real MovieLens dataset"""
    loader = MovieLensLoader('small')  # Use small dataset for testing
    return loader.create_sample_dataset(n_users=500, n_movies=200)

if __name__ == "__main__":
    # Test the loader
    data = load_real_dataset()
    print("\nDataset Statistics:")
    for key, value in data['stats'].items():
        print(f"  {key}: {value}")
    
    print(f"\nSample movies:")
    print(data['movies'][['movieId', 'title_clean', 'genres']].head())
    
    print(f"\nSample ratings:")
    print(data['ratings'][['userId', 'movieId', 'rating']].head())
