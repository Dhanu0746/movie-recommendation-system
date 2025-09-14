class User:
    """User model for the recommendation system"""
    
    def __init__(self, user_id, name, email=None):
        self.user_id = user_id
        self.name = name
        self.email = email
        self.ratings = {}  # {item_id: rating}
        self.preferences = {}  # For content-based filtering
    
    def add_rating(self, item_id, rating):
        """Add a rating for an item"""
        if 1 <= rating <= 5:
            self.ratings[item_id] = rating
        else:
            raise ValueError("Rating must be between 1 and 5")
    
    def get_rating(self, item_id):
        """Get rating for an item"""
        return self.ratings.get(item_id, None)
    
    def get_rated_items(self):
        """Get list of items this user has rated"""
        return list(self.ratings.keys())
    
    def get_average_rating(self):
        """Get average rating given by this user"""
        if not self.ratings:
            return 0
        return sum(self.ratings.values()) / len(self.ratings)
    
    def to_dict(self):
        """Convert user to dictionary for JSON serialization"""
        return {
            'user_id': self.user_id,
            'name': self.name,
            'email': self.email,
            'ratings': self.ratings,
            'average_rating': self.get_average_rating()
        }
