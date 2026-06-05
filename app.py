"""
Movie Recommendation System – Flask Application (Unified SQL Edition)
======================================================================
Provides a REST API + web UI for a hybrid AI recommendation engine that
combines:
  • Collaborative Filtering (user-based + item-based cosine similarity)
  • SVD Matrix Factorisation (personalised user embeddings)
  • Content-Based Filtering with TF-IDF
  • NLP Semantic Search
  • Sentiment-Aware Scoring (VADER over dynamic SQLite reviews)
  • Explainable AI reasons
  • Trending / Popularity Engine (Bayesian Average)
  • Cold-Start handling via popularity fallback
  • SQLite database persistence (SQLAlchemy)
  • Dynamic seeding & MovieLens model training pipeline
"""

import os
import math
from typing import Dict, List
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ── Database and Model imports ────────────────────────────────────────────
from database import db
from models import User, Item, Rating, Review

# ── ML engine imports ─────────────────────────────────────────────────────
from algorithms.collaborative import CollaborativeFiltering
from algorithms.content_based import ContentBasedFiltering
from algorithms.svd_model import SVDModel
from algorithms.sentiment import SentimentEngine
from data.movielens_loader import load_real_dataset

# ── App setup ─────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Database configuration
db_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'cineai.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# ── ML engines ────────────────────────────────────────────────────────────
collaborative_engine = CollaborativeFiltering()
content_engine = ContentBasedFiltering()
svd_engine = SVDModel(n_factors=20)
sentiment_engine = SentimentEngine()


# ══════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════

def _ratings_data() -> List[Dict]:
    all_ratings = Rating.query.all()
    return [{'user_id': r.user_id, 'item_id': r.item_id, 'rating': r.rating}
            for r in all_ratings]


def _items_data() -> List[Dict]:
    all_items = Item.query.all()
    return [item.to_dict() for item in all_items]


def _retrain_all():
    """Re-fit every ML engine with the current database rating + item data."""
    rd = _ratings_data()
    if rd:
        collaborative_engine.fit(rd)
        svd_engine.fit(rd)
    content_engine.fit(_items_data())


def _popularity_scores() -> Dict[int, float]:
    """
    Score every movie by a weighted mix of average rating and number of votes.
    Uses a Bayesian average so movies with few votes don't dominate.
    """
    all_ratings = Rating.query.all()
    if not all_ratings:
        return {}

    from collections import defaultdict
    vote_sum = defaultdict(float)
    vote_cnt = defaultdict(int)
    for r in all_ratings:
        vote_sum[r.item_id] += r.rating
        vote_cnt[r.item_id] += 1

    global_mean = sum(vote_sum.values()) / sum(vote_cnt.values())
    m = 3  # minimum votes required for full confidence

    scores = {}
    all_items = Item.query.all()
    for item in all_items:
        iid = item.item_id
        n = vote_cnt.get(iid, 0)
        v = vote_sum.get(iid, 0.0)
        # Bayesian average
        bayes = (n / (n + m)) * (v / n if n else global_mean) + \
                (m / (n + m)) * global_mean
        # Sentiment boost
        boost = sentiment_engine.get_sentiment_boost(iid)
        scores[iid] = round(bayes * boost, 4)
    return scores


def _enrich_rec(rec_list: List[tuple], user_ratings: Dict, rec_type: str) -> List[Dict]:
    """
    Take a list of (item_id, score) tuples and return enriched dicts with
    item details, Explainable AI reason, and sentiment data from database.
    """
    enriched = []
    for item_id, score in rec_list:
        item = Item.query.get(item_id)
        if not item:
            continue
        reason = content_engine.explain_recommendation(item_id, user_ratings)
        sentiment = sentiment_engine.analyse_movie(item_id)
        enriched.append({
            'item_id':   item_id,
            'score':     round(float(score), 4),
            'type':      rec_type,
            'reason':    reason,
            'sentiment': sentiment,
            'item':      item.to_dict(),
        })
    return enriched


# ══════════════════════════════════════════════════════════════════════════
# Sample data loader
# ══════════════════════════════════════════════════════════════════════════

def load_sample_data():
    """Load sample data if database is empty."""
    if User.query.first() is not None:
        print("Database already has data. Skipping sample data load.")
        return

    print("Loading sample data into SQLite database...")
    sample_users = [
        User(user_id=1, name="Alice",   email="alice@example.com"),
        User(user_id=2, name="Bob",     email="bob@example.com"),
        User(user_id=3, name="Charlie", email="charlie@example.com"),
        User(user_id=4, name="Diana",   email="diana@example.com"),
        User(user_id=5, name="Eve",     email="eve@example.com"),
    ]
    for u in sample_users:
        db.session.add(u)

    sample_movies = [
        Item(item_id=1,  title="The Matrix",        genre="Sci-Fi",    year=1999, director="The Wachowskis",
             description="A computer hacker learns about the true nature of reality and his role in the war against its controllers"),
        Item(item_id=2,  title="Inception",         genre="Sci-Fi",    year=2010, director="Christopher Nolan",
             description="A thief who steals corporate secrets through the use of dream-sharing technology"),
        Item(item_id=3,  title="The Dark Knight",   genre="Action",    year=2008, director="Christopher Nolan",
             description="Batman faces the Joker a criminal mastermind who seeks to undermine the people of Gotham City"),
        Item(item_id=4,  title="Pulp Fiction",      genre="Crime",     year=1994, director="Quentin Tarantino",
             description="The lives of two mob hitmen a boxer and a gangster intertwine in four tales of violence and redemption"),
        Item(item_id=5,  title="Forrest Gump",      genre="Drama",     year=1994, director="Robert Zemeckis",
             description="The life of a simple man who unwittingly influences several major historical events in 20th-century America"),
        Item(item_id=6,  title="The Godfather",     genre="Crime",     year=1972, director="Francis Ford Coppola",
             description="The aging patriarch of an organized crime dynasty transfers control to his reluctant son"),
        Item(item_id=7,  title="Titanic",           genre="Romance",   year=1997, director="James Cameron",
             description="A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious ill-fated Titanic"),
        Item(item_id=8,  title="Avatar",            genre="Sci-Fi",    year=2009, director="James Cameron",
             description="A paraplegic marine dispatched to the moon Pandora on a unique mission becomes torn between following orders and protecting the alien world"),
        Item(item_id=9,  title="Star Wars",         genre="Sci-Fi",    year=1977, director="George Lucas",
             description="Luke Skywalker joins forces with a Jedi Knight pilot and two droids to save the galaxy from the Empire"),
        Item(item_id=10, title="Jurassic Park",     genre="Adventure", year=1993, director="Steven Spielberg",
             description="During a preview tour a theme park suffers a major power breakdown that allows its cloned dinosaur exhibits to run amok"),
        Item(item_id=11, title="The Shawshank Redemption", genre="Drama", year=1994, director="Frank Darabont",
             description="Two imprisoned men bond over a number of years finding solace and eventual redemption through acts of common decency"),
        Item(item_id=12, title="The Lord of the Rings", genre="Fantasy", year=2001, director="Peter Jackson",
             description="A meek Hobbit and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth"),
        Item(item_id=13, title="Interstellar",      genre="Sci-Fi",    year=2014, director="Christopher Nolan",
             description="A team of explorers travel through a wormhole in space in an attempt to ensure humanity survival"),
        Item(item_id=14, title="Gravity",           genre="Sci-Fi",    year=2013, director="Alfonso Cuarón",
             description="Two astronauts work together to survive after an accident leaves them stranded in space"),
        Item(item_id=15, title="The Silence of the Lambs", genre="Thriller", year=1991, director="Jonathan Demme",
             description="A young FBI cadet must receive the help of an incarcerated cannibalistic serial killer to catch another serial killer"),
    ]
    for m in sample_movies:
        db.session.add(m)

    sample_ratings = [
        (1, 1, 5), (1, 2, 4), (1, 3, 5), (1, 6, 4), (1, 9, 5),
        (2, 1, 4), (2, 2, 5), (2, 4, 4), (2, 7, 3), (2, 10, 4),
        (3, 2, 3), (3, 3, 4), (3, 5, 5), (3, 8, 4), (3, 9, 3),
        (4, 1, 5), (4, 3, 4), (4, 6, 5), (4, 7, 4), (4, 9, 4),
        (5, 2, 4), (5, 4, 5), (5, 5, 4), (5, 8, 3), (5, 10, 4),
        (1, 13, 5), (2, 13, 4), (3, 14, 4), (4, 12, 5), (5, 11, 5),
    ]
    for user_id, item_id, rating_val in sample_ratings:
        r = Rating(user_id=user_id, item_id=item_id, rating=rating_val)
        db.session.add(r)

    db.session.commit()


# ══════════════════════════════════════════════════════════════════════════
# Routes – Pages
# ══════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


# ══════════════════════════════════════════════════════════════════════════
# Routes – Users
# ══════════════════════════════════════════════════════════════════════════

@app.route('/api/users', methods=['GET'])
def get_users():
    all_users = User.query.all()
    return jsonify([u.to_dict() for u in all_users])


@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify(user.to_dict())


# ══════════════════════════════════════════════════════════════════════════
# Routes – Movies
# ══════════════════════════════════════════════════════════════════════════

@app.route('/api/items', methods=['GET'])
def get_items():
    all_items = Item.query.all()
    item_list = [item.to_dict() for item in all_items]
    for itm in item_list:
        itm['sentiment'] = sentiment_engine.analyse_movie(itm['item_id'])
    return jsonify(item_list)


@app.route('/api/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    item = Item.query.get(item_id)
    if not item:
        return jsonify({'error': 'Movie not found'}), 404
    data = item.to_dict()
    data['sentiment'] = sentiment_engine.analyse_movie(item_id)
    data['similar'] = []
    for sim_id, sim_score in content_engine.get_similar_items(item_id, 5):
        sim_item = Item.query.get(sim_id)
        if sim_item:
            d = sim_item.to_dict()
            d['similarity'] = round(sim_score, 4)
            data['similar'].append(d)
    return jsonify(data)


# ══════════════════════════════════════════════════════════════════════════
# Routes – Ratings
# ══════════════════════════════════════════════════════════════════════════

@app.route('/api/ratings', methods=['GET'])
def get_ratings():
    all_ratings = Rating.query.all()
    return jsonify([r.to_dict() for r in all_ratings])


@app.route('/api/ratings', methods=['POST'])
def add_rating():
    data = request.get_json()
    user_id = data.get('user_id')
    item_id = data.get('item_id')
    rating_val = data.get('rating')

    if not all([user_id, item_id, rating_val]):
        return jsonify({'error': 'Missing required fields'}), 400

    user = User.query.get(user_id)
    item = Item.query.get(item_id)

    if not user or not item:
        return jsonify({'error': 'Invalid user_id or item_id'}), 400

    try:
        user.add_rating(item_id, rating_val)
        db.session.commit()
        # Retrain models with new database data
        _retrain_all()
        
        # Query recently saved rating object to return
        saved_rating = Rating.query.filter_by(user_id=user_id, item_id=item_id).first()
        return jsonify(saved_rating.to_dict() if saved_rating else {}), 201
    except ValueError as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


# ══════════════════════════════════════════════════════════════════════════
# Routes – Recommendations
# ══════════════════════════════════════════════════════════════════════════

@app.route('/api/recommendations/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    """
    GET /api/recommendations/<user_id>?type=hybrid&n=8
    """
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    rec_type    = request.args.get('type', 'hybrid')
    n_recs      = int(request.args.get('n', 8))
    user_ratings = user.ratings  # Calls custom SQL property

    # ── Cold-start: no ratings → popularity fallback ──────────────────
    if not user_ratings:
        pop = _popularity_scores()
        pop_sorted = sorted(pop.items(), key=lambda x: -x[1])[:n_recs]
        enriched = _enrich_rec(pop_sorted, {}, 'popular')
        return jsonify({'user_id': user_id, 'recommendations': enriched,
                        'type': 'popular', 'cold_start': True})

    # ── Route by type ─────────────────────────────────────────────────
    if rec_type == 'collaborative':
        raw = collaborative_engine.get_user_based_recommendations(user_id, n_recs)

    elif rec_type == 'content':
        raw = content_engine.get_user_recommendations(user_ratings, n_recs)

    elif rec_type == 'svd':
        raw = svd_engine.get_recommendations(user_id, list(user_ratings.keys()), n_recs)

    elif rec_type == 'popular':
        pop = _popularity_scores()
        raw = sorted(pop.items(), key=lambda x: -x[1])[:n_recs]

    else:  # hybrid (default) – blend SVD + collaborative + content + sentiment
        svd_recs   = dict(svd_engine.get_recommendations(
                          user_id, list(user_ratings.keys()), n_recs * 2))
        collab_recs = dict(collaborative_engine.get_hybrid_recommendations(
                          user_id, n_recs * 2))

        # Merge SVD and collaborative scores (60 / 40)
        all_cf = set(svd_recs) | set(collab_recs)
        merged_cf = {iid: 0.6 * svd_recs.get(iid, 0) +
                          0.4 * collab_recs.get(iid, 0)
                     for iid in all_cf}

        # Normalise merged CF scores to [0, 1]
        max_cf = max(merged_cf.values(), default=1) or 1
        merged_cf = {iid: s / max_cf for iid, s in merged_cf.items()}

        # Final hybrid with content
        raw = content_engine.get_hybrid_recommendations(
            user_ratings, merged_cf, n_recs, content_weight=0.3)

        # Apply sentiment boost
        boosted = [(iid, score * sentiment_engine.get_sentiment_boost(iid))
                   for iid, score in raw]
        boosted.sort(key=lambda x: -x[1])
        raw = boosted[:n_recs]

    enriched = _enrich_rec(raw, user_ratings, rec_type)
    return jsonify({'user_id': user_id, 'recommendations': enriched,
                    'type': rec_type})


# ══════════════════════════════════════════════════════════════════════════
# Routes – Similar movies
# ══════════════════════════════════════════════════════════════════════════

@app.route('/api/similar/<int:item_id>', methods=['GET'])
def get_similar_items(item_id):
    item = Item.query.get(item_id)
    if not item:
        return jsonify({'error': 'Item not found'}), 404
    n_similar = int(request.args.get('n', 5))
    similar = content_engine.get_similar_items(item_id, n_similar)
    results = []
    for sim_id, sim_score in similar:
        sim_item = Item.query.get(sim_id)
        if sim_item:
            d = sim_item.to_dict()
            d['similarity'] = round(sim_score, 4)
            d['sentiment']  = sentiment_engine.analyse_movie(sim_id)
            results.append(d)
    return jsonify({'item_id': item_id, 'similar_items': results})


# ══════════════════════════════════════════════════════════════════════════
# Routes – NLP Semantic Search
# ══════════════════════════════════════════════════════════════════════════

@app.route('/api/search', methods=['GET'])
def search_movies():
    """
    GET /api/search?q=space+sci-fi&n=8
    """
    query = request.args.get('q', '').strip()
    n     = int(request.args.get('n', 8))

    if not query:
        return jsonify({'error': 'Query parameter "q" is required'}), 400

    results = content_engine.semantic_search(query, n)
    enriched = []
    for item_id, score in results:
        item = Item.query.get(item_id)
        if item:
            d = item.to_dict()
            d['relevance'] = round(score, 4)
            d['sentiment'] = sentiment_engine.analyse_movie(item_id)
            enriched.append(d)

    return jsonify({'query': query, 'results': enriched})


# ══════════════════════════════════════════════════════════════════════════
# Routes – Trending & Popularity
# ══════════════════════════════════════════════════════════════════════════

@app.route('/api/trending', methods=['GET'])
def get_trending():
    n   = int(request.args.get('n', 10))
    pop = _popularity_scores()
    top = sorted(pop.items(), key=lambda x: -x[1])[:n]

    result = []
    for item_id, score in top:
        item = Item.query.get(item_id)
        if item:
            d = item.to_dict()
            d['popularity_score'] = score
            d['sentiment']        = sentiment_engine.analyse_movie(item_id)
            result.append(d)

    return jsonify({'trending': result})


@app.route('/api/trending/genre', methods=['GET'])
def get_trending_by_genre():
    """
    GET /api/trending/genre?genre=Sci-Fi&n=5
    """
    genre = request.args.get('genre', '').strip()
    n     = int(request.args.get('n', 5))

    pop = _popularity_scores()
    filtered = []
    for iid, score in pop.items():
        item = Item.query.get(iid)
        if item and item.genre and item.genre.lower() == genre.lower():
            filtered.append((iid, score))
            
    filtered.sort(key=lambda x: -x[1])
    result = []
    for item_id, score in filtered[:n]:
        item = Item.query.get(item_id)
        if item:
            d = item.to_dict()
            d['popularity_score'] = score
            d['sentiment']        = sentiment_engine.analyse_movie(item_id)
            result.append(d)

    return jsonify({'genre': genre, 'trending': result})


# ══════════════════════════════════════════════════════════════════════════
# Routes – Evaluation metrics
# ══════════════════════════════════════════════════════════════════════════

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """
    GET /api/metrics
    """
    import random
    rd = _ratings_data()
    if len(rd) < 5:
        return jsonify({'error': 'Not enough ratings to evaluate'}), 400

    random.shuffle(rd)
    split = max(1, int(len(rd) * 0.2))
    test  = rd[:split]
    train = rd[split:]

    eval_model = SVDModel(n_factors=20)
    eval_model.fit(train)

    basic  = eval_model.evaluate(test)
    at_k   = eval_model.precision_recall_at_k(test, k=5)
    return jsonify({**basic, **at_k, 'test_samples': len(test)})


# ══════════════════════════════════════════════════════════════════════════
# Routes – Stats / Analytics
# ══════════════════════════════════════════════════════════════════════════

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """System-level statistics for the admin dashboard."""
    total_users = User.query.count()
    total_items = Item.query.count()
    total_ratings = Rating.query.count()
    
    avg_rating_row = db.session.query(db.func.avg(Rating.rating)).scalar()
    avg_rating = round(float(avg_rating_row), 2) if avg_rating_row is not None else 0.0

    # Genre distribution
    genre_counts = {}
    all_genres = db.session.query(Item.genre).all()
    for (g,) in all_genres:
        if g:
            genre_counts[g] = genre_counts.get(g, 0) + 1

    # Most-rated movies
    from sqlalchemy import func
    most_rated = db.session.query(
        Rating.item_id,
        func.count(Rating.id).label('cnt')
    ).group_by(Rating.item_id).order_by(func.count(Rating.id).desc()).limit(5).all()

    top_movies = []
    for item_id, cnt in most_rated:
        item = Item.query.get(item_id)
        if item:
            d = item.to_dict()
            d['rating_count'] = cnt
            top_movies.append(d)

    return jsonify({
        'total_users':    total_users,
        'total_items':    total_items,
        'total_ratings':  total_ratings,
        'average_rating': avg_rating,
        'genre_distribution': genre_counts,
        'most_rated_movies':  top_movies,
    })


# ══════════════════════════════════════════════════════════════════════════
# Routes – Train (MovieLens loader + Seeder)
# ══════════════════════════════════════════════════════════════════════════

@app.route('/api/train', methods=['POST'])
def train_models():
    """Expand the dataset with real MovieLens data and retrain."""
    try:
        print("Loading real MovieLens dataset...")
        dataset = load_real_dataset()
        
        # Load users
        new_users_count = 0
        for user_id in dataset['users']:
            existing_user = User.query.get(user_id)
            if not existing_user:
                db.session.add(User(user_id=user_id, name=f"User_{user_id}", email=f"user_{user_id}@example.com"))
                new_users_count += 1
        
        # Load movies (items)
        new_items_count = 0
        for _, movie in dataset['movies'].iterrows():
            existing_item = Item.query.get(movie['movieId'])
            if not existing_item:
                new_item = Item(
                    item_id=movie['movieId'],
                    title=movie['title_clean'],
                    genre=movie['genres'],
                    year=movie['year'],
                    description=f"Movie from {movie['year']} with genres: {movie['genres']}"
                )
                db.session.add(new_item)
                new_items_count += 1
                
        # Commit users and items first to satisfy foreign key constraints
        db.session.commit()
        
        # Load ratings
        new_ratings_count = 0
        for _, rating in dataset['ratings'].iterrows():
            existing_rating = Rating.query.filter_by(
                user_id=rating['userId'],
                item_id=rating['movieId']
            ).first()
            if not existing_rating:
                new_rating = Rating(
                    user_id=rating['userId'],
                    item_id=rating['movieId'],
                    rating=rating['rating']
                )
                db.session.add(new_rating)
                new_ratings_count += 1
                
        db.session.commit()
        
        # Clear engine caches and retrain
        sentiment_engine._cache.clear()
        _retrain_all()
        
        return jsonify({
            'success': True,
            'message': 'Models retrained on expanded MovieLens dataset',
            'stats': {
                'new_users': new_users_count,
                'new_items': new_items_count,
                'new_ratings': new_ratings_count,
                'total_users': User.query.count(),
                'total_items': Item.query.count(),
                'total_ratings': Rating.query.count()
            }
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/training-status', methods=['GET'])
def get_training_status():
    """Get current training status for front-end query mapping compatibility."""
    return jsonify({
        'trained': True,
        'stats': {
            'total_users': User.query.count(),
            'total_items': Item.query.count(),
            'total_ratings': Rating.query.count()
        },
        'data_source': 'sqlite_database'
    })


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        load_sample_data()
        _retrain_all()
        
    print("CineAI Recommendation System (SQL Persistence Edition) Ready!")
    print("=================================================================")
    app.run(debug=True, host='0.0.0.0', port=5000)
