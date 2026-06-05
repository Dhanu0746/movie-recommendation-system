from database import db
from datetime import datetime

class Review(db.Model):
    """Review model representing a user's movie review for sentiment analysis"""
    __tablename__ = 'reviews'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    item_id = db.Column(db.Integer, db.ForeignKey('items.item_id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=True)
    
    # Relationships
    user = db.relationship('User', back_populates='reviews_rel')
    item = db.relationship('Item', back_populates='reviews_rel')
    
    def __init__(self, user_id, item_id, content, timestamp=None):
        self.user_id = user_id
        self.item_id = item_id
        self.content = content
        self.timestamp = timestamp or datetime.utcnow()
        
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'item_id': self.item_id,
            'content': self.content,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
