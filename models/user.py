from database import db

class User(db.Model):
    """User model for the recommendation system database"""
    __tablename__ = 'users'
    
    user_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=True)
    
    # Relationships
    ratings_rel = db.relationship('Rating', back_populates='user', cascade='all, delete-orphan', lazy='dynamic')
    reviews_rel = db.relationship('Review', back_populates='user', cascade='all, delete-orphan', lazy='dynamic')
    
    @property
    def ratings(self):
        """Return a dict of {item_id: rating} to maintain compatibility with existing algorithms"""
        return {r.item_id: r.rating for r in self.ratings_rel.all()}
    
    def add_rating(self, item_id, rating_val):
        """Add or update a rating for an item"""
        if not (1 <= rating_val <= 5):
            raise ValueError("Rating must be between 1 and 5")
        
        from models.rating import Rating
        existing = Rating.query.filter_by(user_id=self.user_id, item_id=item_id).first()
        if existing:
            existing.rating = rating_val
        else:
            new_rating = Rating(user_id=self.user_id, item_id=item_id, rating=rating_val)
            db.session.add(new_rating)
            
    def get_rating(self, item_id):
        """Get rating for an item"""
        from models.rating import Rating
        r = Rating.query.filter_by(user_id=self.user_id, item_id=item_id).first()
        return r.rating if r else None
        
    def get_rated_items(self):
        """Get list of items this user has rated"""
        return [r.item_id for r in self.ratings_rel.all()]
        
    def get_average_rating(self):
        """Get average rating given by this user"""
        all_ratings = self.ratings_rel.all()
        if not all_ratings:
            return 0
        return sum(r.rating for r in all_ratings) / len(all_ratings)
        
    def to_dict(self):
        """Convert user to dictionary for JSON serialization"""
        return {
            'user_id': self.user_id,
            'name': self.name,
            'email': self.email,
            'ratings': self.ratings,
            'average_rating': self.get_average_rating()
        }
