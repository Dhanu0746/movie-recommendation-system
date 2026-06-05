from database import db
from datetime import datetime

class Rating(db.Model):
    """Rating model representing a user's rating of an item in the database"""
    __tablename__ = 'ratings'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    item_id = db.Column(db.Integer, db.ForeignKey('items.item_id'), nullable=False)
    rating = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=True)
    
    # Relationships
    user = db.relationship('User', back_populates='ratings_rel')
    item = db.relationship('Item', back_populates='ratings_rel')
    
    def __init__(self, user_id, item_id, rating, timestamp=None):
        self.user_id = user_id
        self.item_id = item_id
        self.rating = rating
        if timestamp:
            if isinstance(timestamp, str):
                try:
                    self.timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    self.timestamp = datetime.utcnow()
            elif isinstance(timestamp, datetime):
                self.timestamp = timestamp
            else:
                self.timestamp = datetime.utcnow()
        else:
            self.timestamp = datetime.utcnow()
            
    def validate_rating(self):
        """Validate that rating is within acceptable range"""
        return 1 <= self.rating <= 5
        
    def to_dict(self):
        """Convert rating to dictionary for JSON serialization"""
        return {
            'user_id': self.user_id,
            'item_id': self.item_id,
            'rating': self.rating,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
        
    def __str__(self):
        return f"User {self.user_id} rated Item {self.item_id}: {self.rating}/5"
