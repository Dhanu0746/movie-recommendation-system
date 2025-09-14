"""
Sample data generator for the recommendation system
"""

import random
from typing import List, Dict

def generate_sample_users(n_users: int = 20) -> List[Dict]:
    """Generate sample users"""
    first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", 
                   "Ivy", "Jack", "Kate", "Liam", "Maya", "Noah", "Olivia", "Paul", 
                   "Quinn", "Rachel", "Sam", "Tina"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", 
                  "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", 
                  "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]
    
    users = []
    for i in range(1, n_users + 1):
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        users.append({
            'user_id': i,
            'name': f"{first_name} {last_name}",
            'email': f"{first_name.lower()}.{last_name.lower()}@example.com"
        })
    
    return users

def generate_sample_movies(n_movies: int = 50) -> List[Dict]:
    """Generate sample movies"""
    titles = [
        "The Matrix", "Inception", "The Dark Knight", "Pulp Fiction", "Forrest Gump",
        "The Godfather", "Titanic", "Avatar", "Star Wars", "Jurassic Park",
        "The Shawshank Redemption", "The Lord of the Rings", "Fight Club", "Goodfellas",
        "The Silence of the Lambs", "Casablanca", "Citizen Kane", "Gone with the Wind",
        "Lawrence of Arabia", "Vertigo", "Psycho", "The Searchers", "Singin' in the Rain",
        "It's a Wonderful Life", "Sunset Boulevard", "The Bridge on the River Kwai",
        "Some Like It Hot", "All About Eve", "The African Queen", "The Maltese Falcon",
        "The Wizard of Oz", "Double Indemnity", "E.T. the Extra-Terrestrial",
        "A Clockwork Orange", "2001: A Space Odyssey", "Dr. Strangelove", "Apocalypse Now",
        "Taxi Driver", "Jaws", "Rocky", "The Exorcist", "The French Connection",
        "The Godfather Part II", "One Flew Over the Cuckoo's Nest", "The Sting",
        "The Graduate", "West Side Story", "The Sound of Music", "My Fair Lady"
    ]
    
    genres = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", 
              "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery", 
              "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    
    directors = [
        "Christopher Nolan", "Quentin Tarantino", "Martin Scorsese", "Steven Spielberg",
        "Francis Ford Coppola", "Stanley Kubrick", "Alfred Hitchcock", "Orson Welles",
        "Billy Wilder", "John Ford", "Howard Hawks", "Frank Capra", "William Wyler",
        "David Lean", "John Huston", "Elia Kazan", "George Cukor", "Victor Fleming",
        "Michael Curtiz", "John Ford", "Robert Wise", "Vincente Minnelli", "George Stevens",
        "Fred Zinnemann", "Joseph L. Mankiewicz", "John Sturges", "Richard Brooks",
        "Mike Nichols", "Robert Wise", "George Cukor"
    ]
    
    movies = []
    for i in range(1, n_movies + 1):
        title = random.choice(titles)
        genre = random.choice(genres)
        year = random.randint(1950, 2023)
        director = random.choice(directors)
        
        # Generate description
        descriptions = [
            f"A compelling {genre.lower()} film that explores themes of love, loss, and redemption.",
            f"An epic {genre.lower()} adventure that takes viewers on an unforgettable journey.",
            f"A masterful {genre.lower()} that showcases brilliant storytelling and cinematography.",
            f"A thought-provoking {genre.lower()} that challenges conventional narratives.",
            f"A visually stunning {genre.lower()} with exceptional performances and direction."
        ]
        
        movies.append({
            'item_id': i,
            'title': title,
            'genre': genre,
            'year': year,
            'director': director,
            'description': random.choice(descriptions)
        })
    
    return movies

def generate_sample_ratings(users: List[Dict], movies: List[Dict], 
                          rating_density: float = 0.3) -> List[Dict]:
    """Generate sample ratings"""
    ratings = []
    n_ratings = int(len(users) * len(movies) * rating_density)
    
    for _ in range(n_ratings):
        user = random.choice(users)
        movie = random.choice(movies)
        rating = random.randint(1, 5)
        
        # Avoid duplicate ratings
        if not any(r['user_id'] == user['user_id'] and r['item_id'] == movie['item_id'] 
                  for r in ratings):
            ratings.append({
                'user_id': user['user_id'],
                'item_id': movie['item_id'],
                'rating': rating
            })
    
    return ratings

def generate_complete_dataset(n_users: int = 20, n_movies: int = 50, 
                            rating_density: float = 0.3) -> Dict:
    """Generate complete dataset"""
    users = generate_sample_users(n_users)
    movies = generate_sample_movies(n_movies)
    ratings = generate_sample_ratings(users, movies, rating_density)
    
    return {
        'users': users,
        'movies': movies,
        'ratings': ratings
    }

if __name__ == "__main__":
    # Generate and print sample data
    dataset = generate_complete_dataset()
    
    print("Sample Users:")
    for user in dataset['users'][:5]:
        print(f"  {user}")
    
    print("\nSample Movies:")
    for movie in dataset['movies'][:5]:
        print(f"  {movie}")
    
    print("\nSample Ratings:")
    for rating in dataset['ratings'][:10]:
        print(f"  {rating}")
    
    print(f"\nDataset Summary:")
    print(f"  Users: {len(dataset['users'])}")
    print(f"  Movies: {len(dataset['movies'])}")
    print(f"  Ratings: {len(dataset['ratings'])}")
