class Rating:
    """Rating model representing a user's rating of an item"""
    
    def __init__(self, user_id, item_id, rating, timestamp=None):
        self.user_id = user_id
        self.item_id = item_id
        self.rating = rating
        self.timestamp = timestamp
    
    def validate_rating(self):
        """Validate that rating is within acceptable range"""
        return 1 <= self.rating <= 5
    
    def to_dict(self):
        """Convert rating to dictionary for JSON serialization"""
        return {
            'user_id': self.user_id,
            'item_id': self.item_id,
            'rating': self.rating,
            'timestamp': self.timestamp
        }
    
    def __str__(self):
        return f"User {self.user_id} rated Item {self.item_id}: {self.rating}/5"
