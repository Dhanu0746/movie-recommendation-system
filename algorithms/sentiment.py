"""
Sentiment Analysis Engine
--------------------------
Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) – a pure-Python,
lexicon-based analyser that works without any model downloads or API keys.

Falls back to a keyword-based scorer if vaderSentiment is not installed.

Resume line:
  Performed sentiment analysis on user reviews using VADER to extract mood
  profiles and improve recommendation relevance.
"""

from typing import Dict, List

# ── Try importing VADER; fall back gracefully ──────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VADER
    _vader_available = True
except ImportError:
    _vader_available = False


# ── Hard-coded sample reviews (simulates an IMDb/TMDB data pull) ───────────
SAMPLE_REVIEWS: Dict[int, List[str]] = {
    1:  ["Mind-blowing concept, absolutely loved it!",
         "A revolutionary film that changed cinema.",
         "Confusing at first but brilliant overall."],
    2:  ["One of the best films ever made. Nolan is a genius.",
         "Incredible visuals and a twisting storyline.",
         "Overrated and too long, didn't enjoy it."],
    3:  ["Heath Ledger's Joker is iconic – masterpiece.",
         "The best superhero film ever created.",
         "Dark, gritty, and perfectly paced."],
    4:  ["A classic. Tarantino at his finest.",
         "Disjointed story but amazing characters.",
         "Not for everyone, but a masterpiece of style."],
    5:  ["A feel-good movie that stays with you forever.",
         "Tom Hanks delivers a flawless performance.",
         "Simple but deeply moving story."],
    6:  ["The greatest film ever made, period.",
         "Marlon Brando is simply unforgettable.",
         "Slow but incredibly rewarding."],
    7:  ["A beautiful love story – tissues required!",
         "Visually stunning but predictable plot.",
         "Cried my eyes out, wonderful movie."],
    8:  ["Visually spectacular but weak story.",
         "Amazing special effects, shallow characters.",
         "Entertaining but forgettable."],
    9:  ["A timeless classic that defined science fiction.",
         "Incredible world-building and adventure.",
         "Still holds up decades later – legendary."],
    10: ["Spielberg at his creative peak – pure fun.",
         "The dinosaurs are still impressive today.",
         "Thrilling from start to finish."],
}


class SentimentEngine:
    """Analyse movie reviews and expose mood-aware sentiment scores."""

    def __init__(self):
        self._analyser = _VADER() if _vader_available else None
        # Cache so each movie is only processed once
        self._cache: Dict[int, Dict] = {}

    def _generate_and_save_mock_reviews(self, movie) -> List[str]:
        """Dynamically generate a set of reviews for a movie, save to DB, and return them."""
        import random
        from models.review import Review
        from database import db
        from models.user import User

        # Decide how positive/negative based on movie's rating if available, or just random
        avg_rating = movie.get_average_rating() or random.uniform(2.5, 4.5)

        pos_templates = [
            f"Absolutely loved this! A classic in the {movie.genre} genre.",
            f"{movie.title} is a masterpiece. Highly recommend it.",
            f"Brilliant performances and spectacular directing by {movie.director or 'the director'}.",
            "A visual feast with a deep, emotional story.",
            "Loved it from start to finish. Truly incredible."
        ]
        neu_templates = [
            f"An okay {movie.genre} movie. Worth a watch if you have free time.",
            f"Great concept but the execution of {movie.title} was just decent.",
            "Had high hopes. It was okay, but slightly long.",
            "Good directing, but the script felt standard.",
            "Decent watch, though a bit slow in the middle."
        ]
        neg_templates = [
            f"Really did not enjoy this. Far too slow and boring.",
            f"Disappointing movie. {movie.title} is definitely overrated.",
            f"Terrible script. Not even {movie.director or 'the director'} could save this.",
            "Predictable plot and forgettable characters.",
            "Waste of time, I couldn't get into it at all."
        ]

        if avg_rating >= 4.0:
            distribution = [('pos', 0.7), ('neu', 0.2), ('neg', 0.1)]
        elif avg_rating <= 2.5:
            distribution = [('pos', 0.1), ('neu', 0.3), ('neg', 0.6)]
        else:
            distribution = [('pos', 0.4), ('neu', 0.4), ('neg', 0.2)]

        generated_contents = []
        for _ in range(3):
            choice = random.choices(['pos', 'neu', 'neg'], weights=[d[1] for d in distribution])[0]
            if choice == 'pos':
                content = random.choice(pos_templates)
            elif choice == 'neu':
                content = random.choice(neu_templates)
            else:
                content = random.choice(neg_templates)
            generated_contents.append(content)

        user = User.query.first()
        uid = user.user_id if user else 1

        for content in generated_contents:
            rev = Review(user_id=uid, item_id=movie.item_id, content=content)
            db.session.add(rev)
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()

        return generated_contents

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def _score_text(self, text: str) -> float:
        """Return a compound sentiment score in [-1, 1]."""
        if self._analyser:
            return self._analyser.polarity_scores(text)['compound']
        # Keyword fallback
        pos = sum(1 for w in ('great','amazing','excellent','love','best','brilliant',
                              'masterpiece','iconic','beautiful','stunning','wonderful',
                              'incredible','legendary','flawless')
                  if w in text.lower())
        neg = sum(1 for w in ('bad','terrible','awful','boring','confusing','weak',
                              'shallow','forgettable','disappointing','overrated','slow')
                  if w in text.lower())
        total = pos + neg
        return (pos - neg) / total if total else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse_movie(self, item_id: int, reviews: List[str] = None) -> Dict:
        """
        Analyse sentiment for a given movie.

        Args:
            item_id: Movie ID.
            reviews: List of review strings.

        Returns:
            {
              'sentiment_score':  float in [-1, 1],
              'sentiment_label':  'Positive' | 'Neutral' | 'Negative',
              'positive_pct':     int (0-100),
              'negative_pct':     int (0-100),
              'review_count':     int,
            }
        """
        if item_id in self._cache:
            return self._cache[item_id]

        if reviews is None:
            try:
                from models.review import Review
                from models.item import Item
                db_reviews = Review.query.filter_by(item_id=item_id).all()
                if db_reviews:
                    reviews = [r.content for r in db_reviews]
                else:
                    movie = Item.query.get(item_id)
                    if movie:
                        reviews = self._generate_and_save_mock_reviews(movie)
                    else:
                        reviews = SAMPLE_REVIEWS.get(item_id, [])
            except Exception:
                reviews = SAMPLE_REVIEWS.get(item_id, [])

        if not reviews:
            result = {
                'sentiment_score': 0.0,
                'sentiment_label': 'Neutral',
                'positive_pct':    50,
                'negative_pct':    50,
                'review_count':    0,
            }
            self._cache[item_id] = result
            return result

        scores = [self._score_text(r) for r in reviews]
        avg = sum(scores) / len(scores)

        positive_count = sum(1 for s in scores if s > 0.05)
        negative_count = sum(1 for s in scores if s < -0.05)
        total = len(scores)

        if avg > 0.05:
            label = 'Positive'
        elif avg < -0.05:
            label = 'Negative'
        else:
            label = 'Neutral'

        result = {
            'sentiment_score': round(avg, 3),
            'sentiment_label': label,
            'positive_pct':    round((positive_count / total) * 100),
            'negative_pct':    round((negative_count / total) * 100),
            'review_count':    total,
        }
        self._cache[item_id] = result
        return result

    def get_sentiment_boost(self, item_id: int) -> float:
        """
        Return a small multiplicative boost (0.8–1.2) based on sentiment,
        used to gently nudge recommendation scores for highly-rated films.
        """
        data = self.analyse_movie(item_id)
        score = data['sentiment_score']
        # Map [-1, 1] → [0.8, 1.2]
        return round(1.0 + 0.2 * score, 4)

    def bulk_analyse(self, item_ids: List[int]) -> Dict[int, Dict]:
        """Analyse all given movie IDs and return a mapping."""
        return {iid: self.analyse_movie(iid) for iid in item_ids}

