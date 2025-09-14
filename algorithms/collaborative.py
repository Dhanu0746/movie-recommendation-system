import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from typing import List, Dict, Tuple

class CollaborativeFiltering:
    """Collaborative filtering recommendation system"""
    
    def __init__(self):
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.ratings_matrix = None
        self.user_ids = None
        self.item_ids = None
    
    def fit(self, ratings_data: List[Dict]):
        """
        Fit the collaborative filtering model
        
        Args:
            ratings_data: List of dictionaries with 'user_id', 'item_id', 'rating'
        """
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(ratings_data)
        
        # Create pivot table (user-item matrix)
        self.ratings_matrix = df.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        )
        
        # Store user and item IDs
        self.user_ids = self.ratings_matrix.index.tolist()
        self.item_ids = self.ratings_matrix.columns.tolist()
        
        # Calculate user similarity matrix
        self.user_similarity_matrix = cosine_similarity(self.ratings_matrix)
        
        # Calculate item similarity matrix
        self.item_similarity_matrix = cosine_similarity(self.ratings_matrix.T)
    
    def get_user_based_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        Get user-based collaborative filtering recommendations
        
        Args:
            user_id: ID of the user to get recommendations for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of tuples (item_id, predicted_rating)
        """
        if user_id not in self.user_ids:
            return []
        
        user_idx = self.user_ids.index(user_id)
        user_ratings = self.ratings_matrix.iloc[user_idx]
        
        # Find similar users
        user_similarities = self.user_similarity_matrix[user_idx]
        
        # Get items not rated by the user
        unrated_items = user_ratings[user_ratings == 0].index.tolist()
        
        predictions = []
        for item_id in unrated_items:
            item_idx = self.item_ids.index(item_id)
            
            # Find users who rated this item
            item_ratings = self.ratings_matrix.iloc[:, item_idx]
            rated_users = item_ratings[item_ratings > 0].index.tolist()
            
            if not rated_users:
                continue
            
            # Calculate weighted average of similar users' ratings
            weighted_sum = 0
            similarity_sum = 0
            
            for rated_user in rated_users:
                rated_user_idx = self.user_ids.index(rated_user)
                similarity = user_similarities[rated_user_idx]
                rating = item_ratings[rated_user]
                
                weighted_sum += similarity * rating
                similarity_sum += abs(similarity)
            
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                predictions.append((item_id, predicted_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def get_item_based_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        Get item-based collaborative filtering recommendations
        
        Args:
            user_id: ID of the user to get recommendations for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of tuples (item_id, predicted_rating)
        """
        if user_id not in self.user_ids:
            return []
        
        user_idx = self.user_ids.index(user_id)
        user_ratings = self.ratings_matrix.iloc[user_idx]
        
        # Get items rated by the user
        rated_items = user_ratings[user_ratings > 0].index.tolist()
        unrated_items = user_ratings[user_ratings == 0].index.tolist()
        
        predictions = []
        for item_id in unrated_items:
            item_idx = self.item_ids.index(item_id)
            item_similarities = self.item_similarity_matrix[item_idx]
            
            weighted_sum = 0
            similarity_sum = 0
            
            for rated_item in rated_items:
                rated_item_idx = self.item_ids.index(rated_item)
                similarity = item_similarities[rated_item_idx]
                rating = user_ratings[rated_item]
                
                weighted_sum += similarity * rating
                similarity_sum += abs(similarity)
            
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                predictions.append((item_id, predicted_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def get_hybrid_recommendations(self, user_id: int, n_recommendations: int = 5, 
                                 user_weight: float = 0.5) -> List[Tuple[int, float]]:
        """
        Get hybrid recommendations combining user-based and item-based filtering
        
        Args:
            user_id: ID of the user to get recommendations for
            n_recommendations: Number of recommendations to return
            user_weight: Weight for user-based recommendations (0-1)
            
        Returns:
            List of tuples (item_id, predicted_rating)
        """
        user_recs = dict(self.get_user_based_recommendations(user_id, n_recommendations * 2))
        item_recs = dict(self.get_item_based_recommendations(user_id, n_recommendations * 2))
        
        # Combine recommendations
        all_items = set(user_recs.keys()) | set(item_recs.keys())
        hybrid_predictions = []
        
        for item_id in all_items:
            user_score = user_recs.get(item_id, 0)
            item_score = item_recs.get(item_id, 0)
            
            # Weighted combination
            hybrid_score = user_weight * user_score + (1 - user_weight) * item_score
            hybrid_predictions.append((item_id, hybrid_score))
        
        # Sort by hybrid score and return top N
        hybrid_predictions.sort(key=lambda x: x[1], reverse=True)
        return hybrid_predictions[:n_recommendations]
