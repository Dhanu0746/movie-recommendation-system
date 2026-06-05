# 🎬 Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive movie recommendation system that implements multiple recommendation algorithms including collaborative filtering, content-based filtering, and hybrid approaches. Built with Flask and featuring a beautiful web interface.

## ✨ Core Machine Learning Features

- **🔀 Hybrid Recommendation Engine:** Combines Collaborative Filtering, Content-Based Filtering, and Popularity-Based Ranking for superior accuracy.
- **🧠 Personalized User Embeddings:** Utilizes advanced matrix factorization techniques (SVD via the Surprise library) to learn complex user preferences.
- **📈 Recommendation Accuracy & Evaluation:** Implements standard ML evaluation metrics (RMSE, Precision@K, Recall@K) to tune performance.
- **🔍 Smart NLP Semantic Search:** Upgrades search functionality to semantic search using natural language via TF-IDF vectorization and cosine similarity.
- **🎭 Sentiment-Aware Recommendations:** Processes IMDb/TMDB reviews through VADER/TextBlob to determine sentiment scores for mood-aware recommendations.
- **⚡ Real-Time Dynamic Updates:** Instantly recalculates and updates user recommendations upon new ratings or interactions.
- **🔥 Trending & Popularity Engine:** A dynamic engine calculating real-time trends such as "Trending Today" and "Most Watched".
- **💡 Explainable AI (XAI):** Enhances user trust by providing clear, human-readable reasons for every recommendation.
- **❄️ Cold Start Problem Resolution:** Implements strategic onboarding and dynamic popularity fallbacks for brand new users.

## 🎨 Premium UI/UX Architecture

- **🏠 Home Page:** Features a dynamic Hero Section, Smart Search Bar, Trending Carousel, and visually distinct Genre Categories.
- **👤 User Dashboard:** Includes Watch History, intuitive Preference Insights, and a user-facing Recommendation Accuracy score.
- **📈 Admin & Analytics Dashboard:** Built with Chart.js/Recharts featuring Dark/Light themes, tracking key metrics like active users and usage analytics.
- **🎬 Movie Detail Page:** Displays high-resolution posters, "AI Recommendation Reason" insights, Similar Content carousels, and sentiment score trends.

## 🎥 Demo

### Live Demo
- **Web Interface**: Beautiful, responsive UI for testing recommendations
- **Real-time Training**: Train on MovieLens dataset with one click
- **Interactive Features**: Add ratings, view recommendations, explore similar movies

### Screenshots
*Add screenshots of your web interface here*

## 🚀 Quick Start

### Prerequisites

**Install Python first:**
1. Go to https://www.python.org/downloads/
2. Download Python 3.11 or 3.12
3. Run installer and **CHECK "Add Python to PATH"**
4. Restart your terminal/command prompt

### Installation

**Option 1: Quick Setup (Windows)**
```bash
# Double-click this file to auto-setup
setup_and_run.bat
```

**Option 2: Manual Setup**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run with sample data
python app.py

# 3. Open browser
# http://localhost:5000
```

**Option 3: Train on Real Data**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train on MovieLens dataset
python run_training.py

# 3. Run enhanced app
python app_with_training.py

# 4. Open browser and click "Train on Real Data"
# http://localhost:5000
```

## 🎯 How It Works

### Recommendation Algorithms

1. **Collaborative Filtering**
   - **User-based**: "Users who liked X also liked Y"
   - **Item-based**: "Items similar to X are Y and Z"
   - Uses cosine similarity to find similar users/items

2. **Content-Based Filtering**
   - Analyzes movie features (title, genre, director, description)
   - Uses TF-IDF vectorization for text features
   - Recommends movies with similar content

3. **Hybrid Approach**
   - Combines collaborative and content-based scores
   - Weighted combination for optimal results

### Sample Data

The system comes with pre-loaded sample data:
- **5 Users**: Alice, Bob, Charlie, Diana, Eve
- **10 Movies**: Popular films like The Matrix, Inception, The Dark Knight, etc.
- **25 Ratings**: Pre-existing user ratings for testing

## 📡 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/users` | Get all users |
| `GET` | `/api/items` | Get all movies |
| `GET` | `/api/ratings` | Get all ratings |
| `POST` | `/api/ratings` | Add a new rating |

### Recommendation Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/recommendations/<user_id>` | Get recommendations for a user |
| `GET` | `/api/similar/<item_id>` | Get movies similar to a given movie |

### Query Parameters

**Recommendations:**
- `type`: `hybrid` (default), `collaborative`, `content`
- `n`: Number of recommendations (default: 5)

**Similar Items:**
- `n`: Number of similar items (default: 5)

### Example API Calls

```bash
# Get hybrid recommendations for user 1
curl "http://localhost:5000/api/recommendations/1?type=hybrid&n=5"

# Get collaborative recommendations
curl "http://localhost:5000/api/recommendations/1?type=collaborative&n=3"

# Get movies similar to The Matrix (ID: 1)
curl "http://localhost:5000/api/similar/1?n=5"

# Add a rating
curl -X POST "http://localhost:5000/api/ratings" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "item_id": 2, "rating": 5}'
```

## 🏗️ Project Structure

```
├── app.py                 # Main Flask application
├── test_system.py         # API testing script
├── requirements.txt       # Python dependencies
├── models/
│   ├── __init__.py
│   ├── user.py           # User model with ratings
│   ├── item.py           # Movie model with features
│   └── rating.py         # Rating model
├── algorithms/
│   ├── __init__.py
│   ├── collaborative.py  # Collaborative filtering algorithms
│   └── content_based.py  # Content-based filtering algorithms
├── static/
│   ├── css/
│   │   └── style.css     # Modern, responsive styling
│   └── js/
│       └── main.js       # Frontend JavaScript
├── templates/
│   └── index.html        # Main web interface
└── data/
    └── sample_data.py    # Sample data generator
```

## 🎨 Web Interface Features

- **👤 User Selection**: Choose from available users
- **📊 User Profile**: View user ratings and statistics
- **🎯 Recommendation Types**: Switch between hybrid, collaborative, and content-based
- **⭐ Add Ratings**: Rate movies to improve recommendations
- **🔍 Similar Movies**: Click on recommendations to find similar movies
- **📈 System Stats**: View overall system statistics
- **📱 Responsive Design**: Works on desktop and mobile devices

## 🔧 Customization

### Adding More Data

1. **Extend sample data:**
```python
# In data/sample_data.py
dataset = generate_complete_dataset(n_users=50, n_movies=100, rating_density=0.4)
```

2. **Add new features:**
```python
# In models/item.py
item.add_feature("budget", 100000000)
item.add_feature("runtime", 120)
```

### Modifying Algorithms

1. **Adjust hybrid weights:**
```python
# In algorithms/content_based.py
hybrid_score = 0.3 * content_score + 0.7 * collab_score
```

2. **Add new similarity metrics:**
```python
# In algorithms/collaborative.py
# Replace cosine_similarity with your preferred metric
```

## 🧪 Testing

Run the test script to verify everything works:

```bash
python test_system.py
```

This will test all API endpoints and show you the results.

## 🚀 Production Deployment

For production deployment:

1. **Use a production WSGI server:**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

2. **Add a database:**
   - Replace in-memory storage with PostgreSQL/MySQL
   - Add database models using SQLAlchemy

3. **Add authentication:**
   - Implement user login/logout
   - Add API authentication tokens

4. **Scale the algorithms:**
   - Use distributed computing for large datasets
   - Implement caching for frequently requested recommendations

## 📚 Dependencies

- **Flask**: Web framework
- **Flask-CORS**: Cross-origin resource sharing
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **scipy**: Scientific computing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Recommending! 🎬✨**
