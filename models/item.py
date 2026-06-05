import json
from database import db

class Item(db.Model):
    """Item model for the recommendation system database (e.g., movies)"""
    __tablename__ = 'items'
    
    item_id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    genre = db.Column(db.String(100), nullable=True)
    year = db.Column(db.Integer, nullable=True)
    director = db.Column(db.String(255), nullable=True)
    description = db.Column(db.Text, nullable=True)
    
    # Store features as a JSON-serialized text field for maximum compatibility
    _features = db.Column('features', db.Text, default='{}')
    
    # Relationships
    ratings_rel = db.relationship('Rating', back_populates='item', cascade='all, delete-orphan', lazy='dynamic')
    reviews_rel = db.relationship('Review', back_populates='item', cascade='all, delete-orphan', lazy='dynamic')
    
    @property
    def features(self):
        try:
            return json.loads(self._features or '{}')
        except Exception:
            return {}
            
    @features.setter
    def features(self, val):
        self._features = json.dumps(val or {})
        
    @property
    def ratings(self):
        """Return a dict of {user_id: rating} to maintain compatibility with existing algorithms"""
        return {r.user_id: r.rating for r in self.ratings_rel.all()}
        
    def add_rating(self, user_id, rating_val):
        """Add a rating from a user"""
        if not (1 <= rating_val <= 5):
            raise ValueError("Rating must be between 1 and 5")
            
        from models.rating import Rating
        existing = Rating.query.filter_by(user_id=user_id, item_id=self.item_id).first()
        if existing:
            existing.rating = rating_val
        else:
            new_rating = Rating(user_id=user_id, item_id=self.item_id, rating=rating_val)
            db.session.add(new_rating)
            
    def get_rating(self, user_id):
        """Get rating from a specific user"""
        from models.rating import Rating
        r = Rating.query.filter_by(user_id=user_id, item_id=self.item_id).first()
        return r.rating if r else None
        
    def get_ratings(self):
        """Get all ratings for this item"""
        return self.ratings
        
    def get_average_rating(self):
        """Get average rating for this item"""
        all_ratings = self.ratings_rel.all()
        if not all_ratings:
            return 0
        return sum(r.rating for r in all_ratings) / len(all_ratings)
        
    def get_rating_count(self):
        """Get number of ratings for this item"""
        return self.ratings_rel.count()
        
    def add_feature(self, feature_name, feature_value):
        """Add a feature for content-based filtering"""
        feats = self.features
        feats[feature_name] = feature_value
        self.features = feats
        
    def get_features(self):
        """Get all features of this item"""
        return self.features
        
    def to_dict(self):
        """Convert item to dictionary for JSON serialization"""
        return {
            'item_id': self.item_id,
            'title': self.title,
            'genre': self.genre,
            'year': self.year,
            'director': self.director,
            'description': self.description,
            'features': self.features,
            'average_rating': self.get_average_rating(),
            'rating_count': self.get_rating_count()
        }
