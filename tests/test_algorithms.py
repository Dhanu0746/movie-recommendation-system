import pytest
import numpy as np
from algorithms.collaborative import CollaborativeFiltering
from algorithms.content_based import ContentBasedFiltering
from algorithms.svd_model import SVDModel
from algorithms.sentiment import SentimentEngine

def test_collaborative_filtering():
    ratings_data = [
        {'user_id': 1, 'item_id': 101, 'rating': 5},
        {'user_id': 1, 'item_id': 102, 'rating': 4},
        {'user_id': 2, 'item_id': 101, 'rating': 5},
        {'user_id': 2, 'item_id': 103, 'rating': 3},
        {'user_id': 3, 'item_id': 102, 'rating': 4},
        {'user_id': 3, 'item_id': 103, 'rating': 4},
    ]
    cf = CollaborativeFiltering()
    cf.fit(ratings_data)
    
    assert 1 in cf.user_ids
    assert 101 in cf.item_ids
    
    # User 1 hasn't rated 103. Let's get predictions
    recs = cf.get_user_based_recommendations(user_id=1, n_recommendations=2)
    assert len(recs) > 0
    assert recs[0][0] == 103  # item 103 is the only unrated item
    assert recs[0][1] > 0

def test_content_based_filtering():
    items_data = [
        {'item_id': 1, 'title': 'The Matrix', 'genre': 'Sci-Fi', 'director': 'Wachowskis', 'description': 'Virtual reality war against machines.'},
        {'item_id': 2, 'title': 'Inception', 'genre': 'Sci-Fi', 'director': 'Christopher Nolan', 'description': 'Dream sharing corporate espionage.'},
        {'item_id': 3, 'title': 'The Godfather', 'genre': 'Crime', 'director': 'Francis Ford Coppola', 'description': 'Italian-American crime family patriarch.'},
    ]
    cbf = ContentBasedFiltering()
    cbf.fit(items_data)
    
    assert len(cbf.item_ids) == 3
    
    # Test similar items: Matrix and Inception should be more similar than Godfather
    sim_items = cbf.get_similar_items(1, n_similar=2)
    assert sim_items[0][0] == 2  # Inception
    assert sim_items[0][1] > sim_items[1][1]  # Inception score > Godfather score
    
    # Test semantic search
    search_results = cbf.semantic_search("dream heist sci-fi")
    assert len(search_results) > 0
    assert search_results[0][0] == 2  # Inception

def test_svd_model():
    ratings_data = [
        {'user_id': 1, 'item_id': 1, 'rating': 5.0},
        {'user_id': 1, 'item_id': 2, 'rating': 4.0},
        {'user_id': 2, 'item_id': 1, 'rating': 5.0},
        {'user_id': 2, 'item_id': 3, 'rating': 3.0},
        {'user_id': 3, 'item_id': 2, 'rating': 4.0},
        {'user_id': 3, 'item_id': 3, 'rating': 4.0},
    ]
    svd = SVDModel(n_factors=2)
    svd.fit(ratings_data)
    
    assert svd.is_fitted
    pred = svd.predict(1, 3)
    assert 1.0 <= pred <= 5.0
    
    # Test metrics computation
    metrics = svd.evaluate([{'user_id': 1, 'item_id': 3, 'rating': 4.0}])
    assert 'rmse' in metrics
    assert 'mae' in metrics

def test_sentiment_engine():
    engine = SentimentEngine()
    
    # Since VADER compound score is in [-1, 1], let's check basic polarity
    pos_res = engine.analyse_movie(1, reviews=["Absolutely amazing movie, loved the acting!", "Incredible plot!"])
    assert pos_res['sentiment_label'] == 'Positive'
    assert pos_res['sentiment_score'] > 0.05
    
    neg_res = engine.analyse_movie(2, reviews=["Terrible and waste of time.", "Boring narrative."])
    assert neg_res['sentiment_label'] == 'Negative'
    assert neg_res['sentiment_score'] < -0.05
