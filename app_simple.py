"""
Simplified recommendation system that works with basic packages only
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
from typing import Dict, List
import math

# Import our models and algorithms
from models.user import User
from models.item import Item
from models.rating import Rating

app = Flask(__name__)
CORS(app)

# In-memory storage
users = {}
items = {}
ratings = []

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(a * a for a in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    return dot_product / (magnitude1 * magnitude2)

class SimpleCollaborativeFiltering:
    """Simplified collaborative filtering without scikit-learn"""
    
    def __init__(self):
        self.user_similarity_matrix = None
        self.ratings_matrix = None
        self.user_ids = None
        self.item_ids = None
    
    def fit(self, ratings_data: List[Dict]):
        """Fit the collaborative filtering model"""
        # Create user-item matrix
        user_ids = list(set(r['user_id'] for r in ratings_data))
        item_ids = list(set(r['item_id'] for r in ratings_data))
        
        self.user_ids = user_ids
        self.item_ids = item_ids
        
        # Create ratings matrix
        self.ratings_matrix = {}
        for rating in ratings_data:
            user_idx = user_ids.index(rating['user_id'])
            item_idx = item_ids.index(rating['item_id'])
            self.ratings_matrix[(user_idx, item_idx)] = rating['rating']
        
        # Calculate user similarity matrix
        self.user_similarity_matrix = {}
        for i, user1 in enumerate(user_ids):
            for j, user2 in enumerate(user_ids):
                if i != j:
                    similarity = self._calculate_user_similarity(i, j)
                    self.user_similarity_matrix[(i, j)] = similarity
    
    def _calculate_user_similarity(self, user1_idx, user2_idx):
        """Calculate similarity between two users"""
        user1_ratings = []
        user2_ratings = []
        
        for item_idx in range(len(self.item_ids)):
            rating1 = self.ratings_matrix.get((user1_idx, item_idx), 0)
            rating2 = self.ratings_matrix.get((user2_idx, item_idx), 0)
            
            if rating1 > 0 and rating2 > 0:  # Both users rated this item
                user1_ratings.append(rating1)
                user2_ratings.append(rating2)
        
        if len(user1_ratings) < 2:
            return 0
        
        return cosine_similarity(user1_ratings, user2_ratings)
    
    def get_user_based_recommendations(self, user_id: int, n_recommendations: int = 5):
        """Get user-based recommendations"""
        if user_id not in self.user_ids:
            return []
        
        user_idx = self.user_ids.index(user_id)
        predictions = []
        
        # Get items not rated by the user
        user_rated_items = set()
        for item_idx in range(len(self.item_ids)):
            if self.ratings_matrix.get((user_idx, item_idx), 0) > 0:
                user_rated_items.add(item_idx)
        
        # Predict ratings for unrated items
        for item_idx in range(len(self.item_ids)):
            if item_idx not in user_rated_items:
                predicted_rating = self._predict_rating(user_idx, item_idx)
                if predicted_rating > 0:
                    item_id = self.item_ids[item_idx]
                    predictions.append((item_id, predicted_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def _predict_rating(self, user_idx, item_idx):
        """Predict rating for a user-item pair"""
        weighted_sum = 0
        similarity_sum = 0
        
        for other_user_idx in range(len(self.user_ids)):
            if other_user_idx != user_idx:
                similarity = self.user_similarity_matrix.get((user_idx, other_user_idx), 0)
                rating = self.ratings_matrix.get((other_user_idx, item_idx), 0)
                
                if rating > 0 and similarity > 0:
                    weighted_sum += similarity * rating
                    similarity_sum += abs(similarity)
        
        if similarity_sum > 0:
            return weighted_sum / similarity_sum
        return 0

# Initialize recommendation engine
collaborative_engine = SimpleCollaborativeFiltering()

def load_sample_data():
    """Load sample data for testing"""
    global users, items, ratings
    
    # Sample users
    sample_users = [
        User(1, "Alice", "alice@example.com"),
        User(2, "Bob", "bob@example.com"),
        User(3, "Charlie", "charlie@example.com"),
        User(4, "Diana", "diana@example.com"),
        User(5, "Eve", "eve@example.com")
    ]
    
    for user in sample_users:
        users[user.user_id] = user
    
    # Sample movies
    sample_movies = [
        Item(1, "The Matrix", "Action", 1999, "The Wachowskis", "A computer hacker learns about the true nature of reality"),
        Item(2, "Inception", "Sci-Fi", 2010, "Christopher Nolan", "A thief who steals corporate secrets through dream-sharing technology"),
        Item(3, "The Dark Knight", "Action", 2008, "Christopher Nolan", "Batman faces the Joker in Gotham City"),
        Item(4, "Pulp Fiction", "Crime", 1994, "Quentin Tarantino", "The lives of two mob hitmen, a boxer, and a gangster intertwine"),
        Item(5, "Forrest Gump", "Drama", 1994, "Robert Zemeckis", "The life of a simple man who unwittingly influences several historical events"),
        Item(6, "The Godfather", "Crime", 1972, "Francis Ford Coppola", "The aging patriarch of a crime dynasty transfers control to his reluctant son"),
        Item(7, "Titanic", "Romance", 1997, "James Cameron", "A seventeen-year-old aristocrat falls in love with a kind but poor artist"),
        Item(8, "Avatar", "Sci-Fi", 2009, "James Cameron", "A paraplegic marine dispatched to the moon Pandora"),
        Item(9, "Star Wars", "Sci-Fi", 1977, "George Lucas", "Luke Skywalker joins forces with a Jedi Knight to save the galaxy"),
        Item(10, "Jurassic Park", "Adventure", 1993, "Steven Spielberg", "A theme park with genetically engineered dinosaurs")
    ]
    
    for movie in sample_movies:
        items[movie.item_id] = movie
    
    # Sample ratings
    sample_ratings = [
        (1, 1, 5), (1, 2, 4), (1, 3, 5), (1, 6, 4), (1, 9, 5),
        (2, 1, 4), (2, 2, 5), (2, 4, 4), (2, 7, 3), (2, 10, 4),
        (3, 2, 3), (3, 3, 4), (3, 5, 5), (3, 8, 4), (3, 9, 3),
        (4, 1, 5), (4, 3, 4), (4, 6, 5), (4, 7, 4), (4, 9, 4),
        (5, 2, 4), (5, 4, 5), (5, 5, 4), (5, 8, 3), (5, 10, 4)
    ]
    
    for user_id, item_id, rating in sample_ratings:
        rating_obj = Rating(user_id, item_id, rating)
        ratings.append(rating_obj)
        users[user_id].add_rating(item_id, rating)
        items[item_id].add_rating(user_id, rating)
    
    # Train the recommendation engine
    ratings_data = [{'user_id': r.user_id, 'item_id': r.item_id, 'rating': r.rating} for r in ratings]
    collaborative_engine.fit(ratings_data)

# API Routes

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users"""
    return jsonify([user.to_dict() for user in users.values()])

@app.route('/api/items', methods=['GET'])
def get_items():
    """Get all items"""
    return jsonify([item.to_dict() for item in items.values()])

@app.route('/api/ratings', methods=['GET'])
def get_ratings():
    """Get all ratings"""
    return jsonify([rating.to_dict() for rating in ratings])

@app.route('/api/ratings', methods=['POST'])
def add_rating():
    """Add a new rating"""
    data = request.get_json()
    
    user_id = data.get('user_id')
    item_id = data.get('item_id')
    rating = data.get('rating')
    
    if not all([user_id, item_id, rating]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    if user_id not in users or item_id not in items:
        return jsonify({'error': 'Invalid user_id or item_id'}), 400
    
    try:
        # Add rating
        rating_obj = Rating(user_id, item_id, rating)
        ratings.append(rating_obj)
        users[user_id].add_rating(item_id, rating)
        items[item_id].add_rating(user_id, rating)
        
        # Retrain models with new data
        ratings_data = [{'user_id': r.user_id, 'item_id': r.item_id, 'rating': r.rating} for r in ratings]
        collaborative_engine.fit(ratings_data)
        
        return jsonify(rating_obj.to_dict()), 201
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/recommendations/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    """Get recommendations for a user"""
    if user_id not in users:
        return jsonify({'error': 'User not found'}), 404
    
    n_recs = int(request.args.get('n', 5))
    
    # Get collaborative filtering recommendations
    recs = collaborative_engine.get_user_based_recommendations(user_id, n_recs)
    recommendations = [{'item_id': item_id, 'score': score, 'type': 'collaborative'} 
                      for item_id, score in recs]
    
    # Add item details to recommendations
    for rec in recommendations:
        item_id = rec['item_id']
        if item_id in items:
            rec['item'] = items[item_id].to_dict()
    
    return jsonify({
        'user_id': user_id,
        'recommendations': recommendations,
        'type': 'collaborative'
    })

@app.route('/api/similar/<int:item_id>', methods=['GET'])
def get_similar_items(item_id):
    """Get items similar to a given item (simplified)"""
    if item_id not in items:
        return jsonify({'error': 'Item not found'}), 404
    
    n_similar = int(request.args.get('n', 5))
    
    # Simple similarity based on genre
    target_item = items[item_id]
    similar_items = []
    
    for other_item_id, other_item in items.items():
        if other_item_id != item_id:
            # Simple genre-based similarity
            similarity = 0.1  # Base similarity
            if target_item.genre and other_item.genre:
                if target_item.genre == other_item.genre:
                    similarity = 0.8
                elif any(genre in other_item.genre for genre in target_item.genre.split('|')):
                    similarity = 0.5
            
            similar_items.append({
                'item_id': other_item_id,
                'similarity': similarity,
                'item': other_item.to_dict()
            })
    
    # Sort by similarity and return top N
    similar_items.sort(key=lambda x: x['similarity'], reverse=True)
    similar_items = similar_items[:n_similar]
    
    return jsonify({
        'item_id': item_id,
        'similar_items': similar_items
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    return jsonify({
        'total_users': len(users),
        'total_items': len(items),
        'total_ratings': len(ratings),
        'average_rating': sum(r.rating for r in ratings) / len(ratings) if ratings else 0
    })

if __name__ == '__main__':
    # Load sample data
    load_sample_data()
    
    print("🎬 Simple Recommendation System Ready!")
    print("=" * 50)
    print("📊 Current Status:")
    print(f"  Users: {len(users)}")
    print(f"  Movies: {len(items)}")
    print(f"  Ratings: {len(ratings)}")
    print("\n🌐 Web Interface: http://localhost:5000")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
