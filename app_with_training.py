"""
Enhanced Flask app with real dataset training capabilities
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
from typing import Dict, List
import pandas as pd
import os

# Import our models and algorithms
from models.user import User
from models.item import Item
from models.rating import Rating
from algorithms.collaborative import CollaborativeFiltering
from algorithms.content_based import ContentBasedFiltering
from data.movielens_loader import load_real_dataset
from train_model import ModelTrainer

app = Flask(__name__)
CORS(app)

# Global variables
users = {}
items = {}
ratings = []
collaborative_engine = CollaborativeFiltering()
content_engine = ContentBasedFiltering()
model_trained = False
training_stats = {}

def load_sample_data():
    """Load sample data for initial testing"""
    global users, items, ratings, model_trained
    
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
    
    # Train the recommendation engines
    ratings_data = [{'user_id': r.user_id, 'item_id': r.item_id, 'rating': r.rating} for r in ratings]
    collaborative_engine.fit(ratings_data)
    
    items_data = [item.to_dict() for item in items.values()]
    content_engine.fit(items_data)
    
    model_trained = True

def train_on_real_data():
    """Train models on real MovieLens dataset"""
    global users, items, ratings, collaborative_engine, content_engine, model_trained, training_stats
    
    try:
        print("Loading real MovieLens dataset...")
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
        
        # Update global variables
        users = trainer.users
        items = trainer.items
        ratings = trainer.ratings
        collaborative_engine = trainer.collaborative_model
        content_engine = trainer.content_model
        model_trained = True
        
        # Store training stats
        training_stats = {
            'dataset_size': {
                'users': len(users),
                'items': len(items),
                'ratings': len(ratings)
            },
            'training_time': {
                'collaborative': collab_time,
                'content': content_time
            },
            'performance': metrics
        }
        
        print("✅ Real dataset training completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error training on real data: {e}")
        return False

# API Routes

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/train', methods=['POST'])
def train_models():
    """Train models on real dataset"""
    try:
        success = train_on_real_data()
        if success:
            return jsonify({
                'success': True,
                'message': 'Models trained successfully on real dataset',
                'stats': training_stats
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to train models on real dataset'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/training-status', methods=['GET'])
def get_training_status():
    """Get current training status"""
    return jsonify({
        'trained': model_trained,
        'stats': training_stats,
        'data_source': 'real_dataset' if training_stats else 'sample_data'
    })

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
    
    # Get recommendation type from query parameters
    rec_type = request.args.get('type', 'hybrid')  # hybrid, collaborative, content
    n_recs = int(request.args.get('n', 5))
    
    user = users[user_id]
    user_ratings = user.ratings
    
    recommendations = []
    
    if rec_type == 'collaborative':
        # User-based collaborative filtering
        recs = collaborative_engine.get_user_based_recommendations(user_id, n_recs)
        recommendations = [{'item_id': item_id, 'score': score, 'type': 'collaborative'} 
                          for item_id, score in recs]
    
    elif rec_type == 'content':
        # Content-based filtering
        recs = content_engine.get_user_recommendations(user_ratings, n_recs)
        recommendations = [{'item_id': item_id, 'score': score, 'type': 'content'} 
                          for item_id, score in recs]
    
    else:  # hybrid
        # Get collaborative recommendations
        collab_recs = collaborative_engine.get_hybrid_recommendations(user_id, n_recs)
        collab_scores = {item_id: score for item_id, score in collab_recs}
        
        # Get hybrid recommendations
        recs = content_engine.get_hybrid_recommendations(user_ratings, collab_scores, n_recs)
        recommendations = [{'item_id': item_id, 'score': score, 'type': 'hybrid'} 
                          for item_id, score in recs]
    
    # Add item details to recommendations
    for rec in recommendations:
        item_id = rec['item_id']
        if item_id in items:
            rec['item'] = items[item_id].to_dict()
    
    return jsonify({
        'user_id': user_id,
        'recommendations': recommendations,
        'type': rec_type,
        'model_trained': model_trained
    })

@app.route('/api/similar/<int:item_id>', methods=['GET'])
def get_similar_items(item_id):
    """Get items similar to a given item"""
    if item_id not in items:
        return jsonify({'error': 'Item not found'}), 404
    
    n_similar = int(request.args.get('n', 5))
    similar_items = content_engine.get_similar_items(item_id, n_similar)
    
    recommendations = []
    for similar_item_id, similarity in similar_items:
        if similar_item_id in items:
            recommendations.append({
                'item_id': similar_item_id,
                'similarity': similarity,
                'item': items[similar_item_id].to_dict()
            })
    
    return jsonify({
        'item_id': item_id,
        'similar_items': recommendations
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    return jsonify({
        'total_users': len(users),
        'total_items': len(items),
        'total_ratings': len(ratings),
        'average_rating': sum(r.rating for r in ratings) / len(ratings) if ratings else 0,
        'model_trained': model_trained,
        'training_stats': training_stats
    })

if __name__ == '__main__':
    # Load sample data initially
    print("Loading sample data...")
    load_sample_data()
    
    print("🎬 Enhanced Recommendation System Ready!")
    print("=" * 50)
    print("📊 Current Status:")
    print(f"  Users: {len(users)}")
    print(f"  Movies: {len(items)}")
    print(f"  Ratings: {len(ratings)}")
    print(f"  Model Trained: {model_trained}")
    print("\n🚀 To train on real MovieLens data:")
    print("  POST /api/train")
    print("\n🌐 Web Interface: http://localhost:5000")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
