"""
Content-Based Filtering with NLP Semantic Search
-------------------------------------------------
Implements TF-IDF content-based recommendations and natural-language semantic
search so queries like "space sci-fi" surface Interstellar, Avatar, Star Wars,
etc.

Resume lines:
  - Integrated NLP-based semantic search using TF-IDF vectorization and cosine
    similarity, enabling intuitive, context-aware movie discovery.
  - Built explainable AI recommendation outputs using feature similarity
    reasoning to enhance user trust and system transparency.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional


class ContentBasedFiltering:
    """Content-based filtering recommendation system with NLP semantic search."""

    def __init__(self):
        self.item_features: Optional[Dict] = None
        self.feature_matrix = None
        self.item_similarity_matrix: Optional[np.ndarray] = None
        self.item_ids: Optional[List] = None
        self.vectorizer: Optional[TfidfVectorizer] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, items_data: List[Dict]) -> None:
        """
        Fit the content-based filtering model.

        Args:
            items_data: List of dicts with item information including features.
        """
        self.item_ids = [item['item_id'] for item in items_data]
        self.item_features = {item['item_id']: item for item in items_data}

        feature_vectors = []
        for item in items_data:
            parts = []
            for field in ('title', 'genre', 'director', 'description'):
                val = item.get(field, '')
                if val:
                    # Repeat genre/title so they carry more weight in TF-IDF
                    repeat = 3 if field in ('genre', 'title') else 1
                    parts.extend([str(val)] * repeat)
            feature_vectors.append(' '.join(parts))

        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=2000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.feature_matrix = self.vectorizer.fit_transform(feature_vectors)
        self.item_similarity_matrix = cosine_similarity(self.feature_matrix)

    # ------------------------------------------------------------------
    # Semantic / NLP Search
    # ------------------------------------------------------------------

    def semantic_search(self, query: str, n_results: int = 8) -> List[Tuple]:
        """
        Return movies that best match a free-text natural-language query.

        Vectorises the query with the fitted TF-IDF model and computes cosine
        similarity against every movie in the catalogue.

        Args:
            query:     Natural-language string, e.g. "space sci-fi adventure".
            n_results: Number of results to return.

        Returns:
            List of (item_id, similarity_score) tuples, best match first.
        """
        if self.vectorizer is None or self.feature_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.feature_matrix).flatten()

        results = sorted(
            [(self.item_ids[i], float(sims[i])) for i in range(len(self.item_ids))],
            key=lambda x: x[1],
            reverse=True
        )
        # Filter out near-zero matches
        results = [(iid, s) for iid, s in results if s > 0.01]
        return results[:n_results]

    # ------------------------------------------------------------------
    # Similar items
    # ------------------------------------------------------------------

    def get_similar_items(self, item_id, n_similar: int = 5) -> List[Tuple]:
        """
        Get items similar to a given item using pre-computed cosine similarity.

        Args:
            item_id:  ID of the reference movie.
            n_similar: Number of similar movies to return.

        Returns:
            List of (item_id, similarity_score) tuples.
        """
        if item_id not in self.item_ids:
            return []

        item_idx = self.item_ids.index(item_id)
        similarities = self.item_similarity_matrix[item_idx]

        similar_items = [
            (self.item_ids[i], float(similarities[i]))
            for i in range(len(self.item_ids))
            if i != item_idx
        ]
        similar_items.sort(key=lambda x: x[1], reverse=True)
        return similar_items[:n_similar]

    # ------------------------------------------------------------------
    # Explainable AI reason
    # ------------------------------------------------------------------

    def explain_recommendation(
        self,
        target_item_id,
        user_ratings: Dict,
        top_n_seed: int = 3
    ) -> str:
        """
        Generate a human-readable reason for a recommendation.

        Example output:
          "Recommended because you liked Inception and Sci-Fi movies."

        Args:
            target_item_id: The recommended movie.
            user_ratings:   {item_id: rating} dict for the user.
            top_n_seed:     How many seed movies to mention.

        Returns:
            Explanation string.
        """
        if not self.item_features or target_item_id not in self.item_features:
            return "Recommended based on your viewing history."

        target = self.item_features[target_item_id]
        target_genre = target.get('genre', '')

        # Find the highest-rated seed movies the user watched in the same genre
        same_genre_seeds = []
        for iid, rating in sorted(user_ratings.items(), key=lambda x: -x[1]):
            if iid in self.item_features:
                seed = self.item_features[iid]
                if seed.get('genre', '').lower() == target_genre.lower():
                    same_genre_seeds.append(seed.get('title', f'Movie {iid}'))
                if len(same_genre_seeds) >= top_n_seed:
                    break

        if same_genre_seeds:
            titles = ' and '.join(same_genre_seeds[:2])
            return f"Recommended because you liked {titles} and {target_genre} movies."

        # Fall back to highest-rated movies generally
        top_seeds = []
        for iid, rating in sorted(user_ratings.items(), key=lambda x: -x[1])[:2]:
            if iid in self.item_features:
                top_seeds.append(self.item_features[iid].get('title', f'Movie {iid}'))

        if top_seeds:
            return f"Recommended because you enjoyed {' and '.join(top_seeds)}."

        return f"Recommended based on popular {target_genre} movies."

    # ------------------------------------------------------------------
    # User recommendations (content-based)
    # ------------------------------------------------------------------

    def get_user_recommendations(
        self,
        user_ratings: Dict[int, float],
        n_recommendations: int = 5
    ) -> List[Tuple]:
        """
        Get content-based recommendations for a user based on their ratings.

        Args:
            user_ratings:       {item_id: rating}
            n_recommendations:  Number of recommendations to return.

        Returns:
            List of (item_id, score) tuples.
        """
        if not user_ratings or self.feature_matrix is None:
            return []

        user_profile = np.zeros(self.feature_matrix.shape[1])
        total_weight = 0.0

        for item_id, rating in user_ratings.items():
            if item_id in self.item_ids:
                idx = self.item_ids.index(item_id)
                item_vec = self.feature_matrix[idx].toarray().flatten()
                user_profile += item_vec * float(rating)
                total_weight += float(rating)

        if total_weight > 0:
            user_profile /= total_weight

        sims = cosine_similarity(user_profile.reshape(1, -1), self.feature_matrix).flatten()

        recs = [
            (self.item_ids[i], float(sims[i]))
            for i in range(len(self.item_ids))
            if self.item_ids[i] not in user_ratings
        ]
        recs.sort(key=lambda x: x[1], reverse=True)
        return recs[:n_recommendations]

    # ------------------------------------------------------------------
    # Hybrid recommendations
    # ------------------------------------------------------------------

    def get_hybrid_recommendations(
        self,
        user_ratings: Dict[int, float],
        collaborative_scores: Dict[int, float],
        n_recommendations: int = 5,
        content_weight: float = 0.3
    ) -> List[Tuple]:
        """
        Hybrid: blend content-based scores with collaborative filtering scores.

        Args:
            user_ratings:          {item_id: rating}
            collaborative_scores:  {item_id: score} from collaborative/SVD engine.
            n_recommendations:     Number of results.
            content_weight:        Weight for content-based score (0–1).

        Returns:
            List of (item_id, hybrid_score) tuples.
        """
        content_recs = dict(
            self.get_user_recommendations(user_ratings, n_recommendations * 2)
        )

        all_items = set(content_recs) | set(collaborative_scores)
        hybrid = []
        for item_id in all_items:
            cs = max(0.0, min(1.0, content_recs.get(item_id, 0.0)))
            cf = max(0.0, min(1.0, collaborative_scores.get(item_id, 0.0)))
            hybrid.append((item_id, content_weight * cs + (1 - content_weight) * cf))

        hybrid.sort(key=lambda x: x[1], reverse=True)
        return hybrid[:n_recommendations]
