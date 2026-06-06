# 🎬 CineAI - Hybrid Movie Recommendation System

## 🚀 Live Demo

🌐 Live Application:
https://movie-recommendation-system-ohbc.onrender.com

📂 GitHub Repository:
https://github.com/Dhanu0746/YOUR_REPOSITORY_NAME

---

## 📌 Overview

CineAI is an AI-powered Hybrid Movie Recommendation System built using Flask, SQLite, and Machine Learning techniques. The application combines Collaborative Filtering, Content-Based Filtering, SVD Matrix Factorization, Sentiment Analysis, and Semantic Search to provide personalized movie recommendations.

The system also includes explainable recommendations, trending movie analysis, recommendation evaluation metrics, Docker support, and cloud deployment.

---

## ✨ Features

### 🤖 AI & Machine Learning

- Hybrid Recommendation Engine
- Collaborative Filtering
- Content-Based Filtering
- SVD Matrix Factorization
- Semantic Search using TF-IDF
- Cosine Similarity Matching
- Sentiment-Aware Recommendations
- Explainable AI Recommendations
- Cold Start User Handling

### 📊 Analytics

- Trending Movies Engine
- Popularity-Based Ranking
- Recommendation Evaluation Metrics
  - RMSE
  - Precision@K
  - Recall@K

### 🌐 Backend

- Flask REST APIs
- SQLite Database
- Dynamic Model Retraining
- SQLAlchemy ORM
- JSON API Responses

### ⚙️ DevOps

- Dockerized Deployment
- Docker Compose Support
- GitHub Integration
- Cloud Hosted on Render

---

## 🏗️ System Architecture

```text
User
 │
 ▼
Web Interface
 │
 ▼
Flask REST API
 │
 ├── Collaborative Filtering
 ├── Content-Based Filtering
 ├── SVD Matrix Factorization
 ├── Sentiment Analysis
 └── Semantic Search
 │
 ▼
SQLite Database
 │
 ▼
Personalized Recommendations
```

---

## 🛠️ Tech Stack

### Backend

- Python
- Flask
- Flask-CORS
- SQLAlchemy
- SQLite

### Machine Learning

- Scikit-Learn
- NumPy
- Pandas
- Surprise (SVD)

### NLP

- TF-IDF Vectorization
- Cosine Similarity
- VADER Sentiment Analysis

### DevOps

- Docker
- Docker Compose
- GitHub
- Render

---

## 📂 Project Structure

```text
movie-recommendation-system/
│
├── algorithms/
│   ├── collaborative.py
│   ├── content_based.py
│   ├── sentiment.py
│   └── svd_model.py
│
├── models/
│   ├── user.py
│   ├── item.py
│   ├── rating.py
│   └── review.py
│
├── data/
│   └── movielens_loader.py
│
├── templates/
├── static/
├── tests/
│
├── app.py
├── database.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Dhanu0746/YOUR_REPOSITORY_NAME.git

cd YOUR_REPOSITORY_NAME
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t cineai .
```

### Run Container

```bash
docker run -p 5000:5000 cineai
```

Application:

```text
http://localhost:5000
```

### Docker Compose

```bash
docker-compose up --build
```

Stop:

```bash
docker-compose down
```

---

## 📡 API Endpoints

### Users

```http
GET /api/users
GET /api/users/<user_id>
```

### Movies

```http
GET /api/items
GET /api/items/<item_id>
```

### Ratings

```http
GET /api/ratings
POST /api/ratings
```

### Recommendations

```http
GET /api/recommendations/<user_id>
```

Parameters:

```text
type = hybrid | collaborative | content | svd | popular

n = number of recommendations
```

### Similar Movies

```http
GET /api/similar/<item_id>
```

### Semantic Search

```http
GET /api/search?q=<query>
```

### Trending Movies

```http
GET /api/trending

GET /api/trending/genre?genre=Sci-Fi
```

### Evaluation Metrics

```http
GET /api/metrics
```

### Analytics

```http
GET /api/stats
```

---

## 📈 Recommendation Workflow

```text
User Ratings
      │
      ▼
Data Collection
      │
      ▼
Feature Extraction
      │
      ▼
Collaborative Filtering
      │
      ▼
Content-Based Filtering
      │
      ▼
SVD Matrix Factorization
      │
      ▼
Sentiment Analysis
      │
      ▼
Hybrid Ranking
      │
      ▼
Final Recommendations
```

---

## 🧪 Evaluation Metrics

The recommendation engine supports:

- RMSE (Root Mean Squared Error)
- Precision@K
- Recall@K

These metrics help evaluate recommendation quality and ranking effectiveness.

---

## 🎯 Key Achievements

- Developed a hybrid recommendation engine using Collaborative Filtering, Content-Based Filtering, and SVD Matrix Factorization.
- Implemented semantic movie search using TF-IDF and cosine similarity.
- Integrated sentiment-aware recommendation ranking.
- Built explainable AI recommendations.
- Designed REST APIs using Flask and SQLite.
- Containerized the application using Docker.
- Deployed the application on Render cloud infrastructure.

---



## 🚀 Future Enhancements

- PostgreSQL Integration
- User Authentication & Authorization
- Advanced Deep Learning Recommendation Models
- Real-Time Recommendation Updates
- Recommendation Explainability Dashboard
- Kubernetes Deployment

---

## 👨‍💻 Author

**Dhanu Shree**

GitHub:
https://github.com/Dhanu0746

---

---

⭐ If you found this project useful, consider giving it a star on GitHub.
