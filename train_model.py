"""
Model training and evaluation script for the recommendation system
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import time

# Import our models and algorithms
from models.user import User
from models.item import Item
from models.rating import Rating
from algorithms.collaborative import CollaborativeFiltering
from algorithms.content_based import ContentBasedFiltering
from data.movielens_loader import load_real_dataset

class ModelTrainer:
    """Train and evaluate recommendation models"""
    
    def __init__(self):
        self.collaborative_model = CollaborativeFiltering()
        self.content_model = ContentBasedFiltering()
        self.users = {}
        self.items = {}
        self.ratings = []
        
    def load_data(self, dataset: Dict):
        """Load dataset into our models"""
        print("Loading data into models...")
        
        # Create users
        for user_id in dataset['users']:
            self.users[user_id] = User(user_id, f"User_{user_id}")
        
        # Create items (movies)
        for _, movie in dataset['movies'].iterrows():
            item = Item(
                item_id=movie['movieId'],
                title=movie['title_clean'],
                genre=movie['genres'],
                year=movie['year'],
                description=f"Movie from {movie['year']} with genres: {movie['genres']}"
            )
            self.items[movie['movieId']] = item
        
        # Create ratings
        for _, rating in dataset['ratings'].iterrows():
            rating_obj = Rating(
                user_id=rating['userId'],
                item_id=rating['movieId'],
                rating=rating['rating']
            )
            self.ratings.append(rating_obj)
            
            # Add to user and item objects
            self.users[rating['userId']].add_rating(rating['movieId'], rating['rating'])
            self.items[rating['movieId']].add_rating(rating['userId'], rating['rating'])
        
        print(f"Loaded {len(self.users)} users, {len(self.items)} items, {len(self.ratings)} ratings")
    
    def prepare_training_data(self, test_size: float = 0.2):
        """Split data into training and testing sets"""
        print("Preparing training and testing data...")
        
        # Convert ratings to DataFrame
        ratings_data = []
        for rating in self.ratings:
            ratings_data.append({
                'user_id': rating.user_id,
                'item_id': rating.item_id,
                'rating': rating.rating
            })
        
        ratings_df = pd.DataFrame(ratings_data)
        
        # Split into train and test
        train_ratings, test_ratings = train_test_split(
            ratings_df, 
            test_size=test_size, 
            random_state=42,
            stratify=ratings_df['rating']
        )
        
        print(f"Training set: {len(train_ratings)} ratings")
        print(f"Testing set: {len(test_ratings)} ratings")
        
        return train_ratings, test_ratings
    
    def train_models(self, train_ratings: pd.DataFrame):
        """Train both collaborative and content-based models"""
        print("Training models...")
        
        # Train collaborative filtering
        print("Training collaborative filtering model...")
        start_time = time.time()
        self.collaborative_model.fit(train_ratings.to_dict('records'))
        collab_time = time.time() - start_time
        print(f"Collaborative filtering training completed in {collab_time:.2f} seconds")
        
        # Train content-based filtering
        print("Training content-based filtering model...")
        start_time = time.time()
        items_data = [item.to_dict() for item in self.items.values()]
        self.content_model.fit(items_data)
        content_time = time.time() - start_time
        print(f"Content-based filtering training completed in {content_time:.2f} seconds")
        
        return collab_time, content_time
    
    def evaluate_models(self, test_ratings: pd.DataFrame) -> Dict:
        """Evaluate model performance"""
        print("Evaluating models...")
        
        results = {
            'collaborative': {'predictions': [], 'actual': []},
            'content': {'predictions': [], 'actual': []},
            'hybrid': {'predictions': [], 'actual': []}
        }
        
        # Test on a sample of ratings (for performance)
        test_sample = test_ratings.sample(n=min(1000, len(test_ratings)), random_state=42)
        
        for _, row in test_sample.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            actual_rating = row['rating']
            
            # Get user's ratings for content-based filtering
            user_ratings = self.users[user_id].ratings.copy()
            if item_id in user_ratings:
                del user_ratings[item_id]  # Remove the test rating
            
            try:
                # Collaborative filtering prediction
                collab_recs = self.collaborative_model.get_user_based_recommendations(user_id, 10)
                collab_pred = next((score for item, score in collab_recs if item == item_id), None)
                if collab_pred:
                    results['collaborative']['predictions'].append(collab_pred)
                    results['collaborative']['actual'].append(actual_rating)
                
                # Content-based filtering prediction
                content_recs = self.content_model.get_user_recommendations(user_ratings, 10)
                content_pred = next((score for item, score in content_recs if item == item_id), None)
                if content_pred:
                    results['content']['predictions'].append(content_pred)
                    results['content']['actual'].append(actual_rating)
                
                # Hybrid prediction
                if collab_pred and content_pred:
                    hybrid_pred = 0.5 * collab_pred + 0.5 * content_pred
                    results['hybrid']['predictions'].append(hybrid_pred)
                    results['hybrid']['actual'].append(actual_rating)
                    
            except Exception as e:
                continue  # Skip if prediction fails
        
        # Calculate metrics
        metrics = {}
        for model_name, data in results.items():
            if data['predictions']:
                predictions = np.array(data['predictions'])
                actual = np.array(data['actual'])
                
                mse = mean_squared_error(actual, predictions)
                mae = mean_absolute_error(actual, predictions)
                rmse = np.sqrt(mse)
                
                metrics[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'n_predictions': len(predictions)
                }
        
        return metrics
    
    def generate_recommendations_report(self, n_users: int = 5) -> Dict:
        """Generate recommendations for sample users"""
        print("Generating recommendations report...")
        
        sample_users = list(self.users.keys())[:n_users]
        recommendations_report = {}
        
        for user_id in sample_users:
            user = self.users[user_id]
            
            # Get recommendations from all models
            collab_recs = self.collaborative_model.get_user_based_recommendations(user_id, 5)
            content_recs = self.content_model.get_user_recommendations(user.ratings, 5)
            
            # Hybrid recommendations
            collab_scores = {item: score for item, score in collab_recs}
            hybrid_recs = self.content_model.get_hybrid_recommendations(
                user.ratings, collab_scores, 5
            )
            
            recommendations_report[user_id] = {
                'user_name': user.name,
                'user_ratings': len(user.ratings),
                'collaborative': [
                    {'item_id': item, 'title': self.items[item].title, 'score': score}
                    for item, score in collab_recs
                ],
                'content': [
                    {'item_id': item, 'title': self.items[item].title, 'score': score}
                    for item, score in content_recs
                ],
                'hybrid': [
                    {'item_id': item, 'title': self.items[item].title, 'score': score}
                    for item, score in hybrid_recs
                ]
            }
        
        return recommendations_report
    
    def save_model_results(self, metrics: Dict, recommendations: Dict, filename: str = 'model_results.json'):
        """Save model results to file"""
        import json
        
        results = {
            'metrics': metrics,
            'recommendations': recommendations,
            'dataset_stats': {
                'total_users': len(self.users),
                'total_items': len(self.items),
                'total_ratings': len(self.ratings)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {filename}")

def main():
    """Main training pipeline"""
    print("🎬 Movie Recommendation System - Model Training")
    print("=" * 60)
    
    # Load real dataset
    print("Loading MovieLens dataset...")
    dataset = load_real_dataset()
    
    # Initialize trainer
    trainer = ModelTrainer()
    trainer.load_data(dataset)
    
    # Prepare training data
    train_ratings, test_ratings = trainer.prepare_training_data()
    
    # Train models
    collab_time, content_time = trainer.train_models(train_ratings)
    
    # Evaluate models
    metrics = trainer.evaluate_models(test_ratings)
    
    # Generate recommendations
    recommendations = trainer.generate_recommendations_report()
    
    # Print results
    print("\n📊 Model Performance:")
    print("-" * 40)
    for model_name, metric in metrics.items():
        print(f"\n{model_name.upper()} Model:")
        print(f"  RMSE: {metric['rmse']:.3f}")
        print(f"  MAE:  {metric['mae']:.3f}")
        print(f"  Predictions: {metric['n_predictions']}")
    
    print(f"\n⏱️ Training Times:")
    print(f"  Collaborative Filtering: {collab_time:.2f}s")
    print(f"  Content-Based Filtering: {content_time:.2f}s")
    
    # Save results
    trainer.save_model_results(metrics, recommendations)
    
    print("\n✅ Training completed successfully!")
    print("Results saved to 'model_results.json'")

if __name__ == "__main__":
    main()
