class Item:
    """Item model for the recommendation system (e.g., movies, products)"""
    
    def __init__(self, item_id, title, genre=None, year=None, director=None, description=None):
        self.item_id = item_id
        self.title = title
        self.genre = genre
        self.year = year
        self.director = director
        self.description = description
        self.features = {}  # For content-based filtering
        self.ratings = {}  # {user_id: rating}
    
    def add_rating(self, user_id, rating):
        """Add a rating from a user"""
        if 1 <= rating <= 5:
            self.ratings[user_id] = rating
        else:
            raise ValueError("Rating must be between 1 and 5")
    
    def get_rating(self, user_id):
        """Get rating from a specific user"""
        return self.ratings.get(user_id, None)
    
    def get_ratings(self):
        """Get all ratings for this item"""
        return self.ratings
    
    def get_average_rating(self):
        """Get average rating for this item"""
        if not self.ratings:
            return 0
        return sum(self.ratings.values()) / len(self.ratings)
    
    def get_rating_count(self):
        """Get number of ratings for this item"""
        return len(self.ratings)
    
    def add_feature(self, feature_name, feature_value):
        """Add a feature for content-based filtering"""
        self.features[feature_name] = feature_value
    
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
