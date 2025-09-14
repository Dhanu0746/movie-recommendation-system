import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

class ContentBasedFiltering:
    """Content-based filtering recommendation system"""
    
    def __init__(self):
        self.item_features = None
        self.feature_matrix = None
        self.item_similarity_matrix = None
        self.item_ids = None
        self.vectorizer = None
    
    def fit(self, items_data: List[Dict]):
        """
        Fit the content-based filtering model
        
        Args:
            items_data: List of dictionaries with item information including features
        """
        self.item_ids = [item['item_id'] for item in items_data]
        self.item_features = {item['item_id']: item for item in items_data}
        
        # Create feature vectors for each item
        feature_vectors = []
        
        for item in items_data:
            # Combine text features (title, description, genre, etc.)
            text_features = []
            
            if 'title' in item and item['title']:
                text_features.append(str(item['title']))
            if 'genre' in item and item['genre']:
                text_features.append(str(item['genre']))
            if 'director' in item and item['director']:
                text_features.append(str(item['director']))
            if 'description' in item and item['description']:
                text_features.append(str(item['description']))
            
            # Join all text features
            combined_text = ' '.join(text_features)
            feature_vectors.append(combined_text)
        
        # Use TF-IDF to vectorize text features
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        
        self.feature_matrix = self.vectorizer.fit_transform(feature_vectors)
        
        # Calculate item similarity matrix
        self.item_similarity_matrix = cosine_similarity(self.feature_matrix)
    
    def get_similar_items(self, item_id: int, n_similar: int = 5) -> List[Tuple[int, float]]:
        """
        Get items similar to a given item
        
        Args:
            item_id: ID of the item to find similar items for
            n_similar: Number of similar items to return
            
        Returns:
            List of tuples (item_id, similarity_score)
        """
        if item_id not in self.item_ids:
            return []
        
        item_idx = self.item_ids.index(item_id)
        similarities = self.item_similarity_matrix[item_idx]
        
        # Get top similar items (excluding the item itself)
        similar_items = []
        for i, similarity in enumerate(similarities):
            if i != item_idx:  # Exclude the item itself
                similar_items.append((self.item_ids[i], similarity))
        
        # Sort by similarity and return top N
        similar_items.sort(key=lambda x: x[1], reverse=True)
        return similar_items[:n_similar]
    
    def get_user_recommendations(self, user_ratings: Dict[int, float], 
                               n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        Get content-based recommendations for a user based on their ratings
        
        Args:
            user_ratings: Dictionary of {item_id: rating} for the user
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of tuples (item_id, predicted_rating)
        """
        if not user_ratings:
            return []
        
        # Calculate user profile based on rated items
        user_profile = np.zeros(self.feature_matrix.shape[1])
        total_weight = 0
        
        for item_id, rating in user_ratings.items():
            if item_id in self.item_ids:
                item_idx = self.item_ids.index(item_id)
                item_features = self.feature_matrix[item_idx].toarray().flatten()
                
                # Weight features by user's rating
                user_profile += item_features * rating
                total_weight += rating
        
        if total_weight > 0:
            user_profile /= total_weight
        
        # Calculate similarity between user profile and all items
        user_profile = user_profile.reshape(1, -1)
        similarities = cosine_similarity(user_profile, self.feature_matrix).flatten()
        
        # Get recommendations for unrated items
        recommendations = []
        for i, similarity in enumerate(similarities):
            item_id = self.item_ids[i]
            if item_id not in user_ratings:  # Only recommend unrated items
                recommendations.append((item_id, similarity))
        
        # Sort by similarity and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def get_hybrid_recommendations(self, user_ratings: Dict[int, float], 
                                 collaborative_scores: Dict[int, float],
                                 n_recommendations: int = 5,
                                 content_weight: float = 0.3) -> List[Tuple[int, float]]:
        """
        Get hybrid recommendations combining content-based and collaborative filtering
        
        Args:
            user_ratings: Dictionary of {item_id: rating} for the user
            collaborative_scores: Dictionary of {item_id: score} from collaborative filtering
            n_recommendations: Number of recommendations to return
            content_weight: Weight for content-based recommendations (0-1)
            
        Returns:
            List of tuples (item_id, predicted_rating)
        """
        content_recs = dict(self.get_user_recommendations(user_ratings, n_recommendations * 2))
        
        # Combine content-based and collaborative scores
        all_items = set(content_recs.keys()) | set(collaborative_scores.keys())
        hybrid_predictions = []
        
        for item_id in all_items:
            content_score = content_recs.get(item_id, 0)
            collab_score = collaborative_scores.get(item_id, 0)
            
            # Normalize scores to 0-1 range
            content_score = max(0, min(1, content_score))
            collab_score = max(0, min(1, collab_score))
            
            # Weighted combination
            hybrid_score = content_weight * content_score + (1 - content_weight) * collab_score
            hybrid_predictions.append((item_id, hybrid_score))
        
        # Sort by hybrid score and return top N
        hybrid_predictions.sort(key=lambda x: x[1], reverse=True)
        return hybrid_predictions[:n_recommendations]
