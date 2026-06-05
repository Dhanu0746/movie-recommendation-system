"""
SVD Matrix Factorization for Personalized Recommendations
---------------------------------------------------------
Implements truncated SVD (via scipy) on the user-item rating matrix to produce
latent-factor user and item embeddings.  These embeddings drive the personalized
recommendation scores that the hybrid engine uses.

Resume line:
  Implemented matrix factorization (Truncated SVD) for personalized user
  embedding generation and collaborative recommendation scoring.
"""

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from typing import List, Dict, Tuple


class SVDModel:
    """Truncated SVD matrix-factorisation recommendation model."""

    def __init__(self, n_factors: int = 20):
        """
        Args:
            n_factors: Number of latent factors (rank of approximation).
        """
        self.n_factors = n_factors
        self.user_factors: np.ndarray = None   # U  (users  × k)
        self.sigma: np.ndarray = None          # Σ  (k,)
        self.item_factors: np.ndarray = None   # Vt (k × items)
        self.predicted_ratings: np.ndarray = None
        self.user_ids: List = []
        self.item_ids: List = []
        self.user_means: Dict = {}
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, ratings_data: List[Dict]) -> None:
        """
        Fit the SVD model from a list of rating dicts.

        Args:
            ratings_data: List of {'user_id': …, 'item_id': …, 'rating': …}
        """
        if len(ratings_data) < 3:
            return

        df = pd.DataFrame(ratings_data)

        # Build user-item pivot table (fill missing with 0)
        matrix_df = df.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            fill_value=0
        )

        self.user_ids = matrix_df.index.tolist()
        self.item_ids = matrix_df.columns.tolist()

        matrix = matrix_df.values.astype(float)

        # Mean-centre each user's ratings (improves factorisation quality)
        self.user_means = {uid: np.mean(matrix[i][matrix[i] > 0])
                           for i, uid in enumerate(self.user_ids)}
        for i, uid in enumerate(self.user_ids):
            mean = self.user_means.get(uid, 0)
            matrix[i][matrix[i] > 0] -= mean

        # Clamp k so we never ask for more factors than the matrix allows
        k = min(self.n_factors, min(matrix.shape) - 1)
        if k < 1:
            return

        # Truncated SVD
        U, sigma, Vt = svds(matrix, k=k)

        # svds returns singular values in *ascending* order; reverse so the
        # most important factors come first.
        idx = np.argsort(sigma)[::-1]
        U, sigma, Vt = U[:, idx], sigma[idx], Vt[idx, :]

        self.user_factors = U
        self.sigma = sigma
        self.item_factors = Vt

        # Reconstruct full predicted-rating matrix and add back user means
        self.predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt)
        for i, uid in enumerate(self.user_ids):
            mean = self.user_means.get(uid, 0)
            self.predicted_ratings[i] += mean

        # Clip to valid rating range [1, 5]
        self.predicted_ratings = np.clip(self.predicted_ratings, 1.0, 5.0)
        self.is_fitted = True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, user_id, item_id) -> float:
        """
        Predict the rating a user would give to an item.

        Returns:
            Predicted rating in [1, 5], or the user's mean rating if unknown.
        """
        if not self.is_fitted:
            return 3.0
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return self.user_means.get(user_id, 3.0)

        u_idx = self.user_ids.index(user_id)
        i_idx = self.item_ids.index(item_id)
        return float(self.predicted_ratings[u_idx, i_idx])

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def get_recommendations(
        self,
        user_id,
        rated_item_ids: List,
        n_recommendations: int = 5
    ) -> List[Tuple]:
        """
        Return the top-N unrated items sorted by predicted rating.

        Args:
            user_id:          Target user.
            rated_item_ids:   Items the user has already rated (exclude these).
            n_recommendations: How many to return.

        Returns:
            List of (item_id, predicted_rating) tuples.
        """
        if not self.is_fitted or user_id not in self.user_ids:
            return []

        u_idx = self.user_ids.index(user_id)
        predictions = []

        for i_idx, item_id in enumerate(self.item_ids):
            if item_id not in rated_item_ids:
                score = float(self.predicted_ratings[u_idx, i_idx])
                predictions.append((item_id, score))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

    # ------------------------------------------------------------------
    # Evaluation metrics
    # ------------------------------------------------------------------

    def evaluate(self, test_ratings: List[Dict]) -> Dict[str, float]:
        """
        Compute RMSE and MAE on a held-out test set.

        Args:
            test_ratings: List of {'user_id', 'item_id', 'rating'}

        Returns:
            {'rmse': …, 'mae': …}
        """
        if not self.is_fitted or not test_ratings:
            return {'rmse': 0.0, 'mae': 0.0}

        errors = []
        for r in test_ratings:
            predicted = self.predict(r['user_id'], r['item_id'])
            errors.append(r['rating'] - predicted)

        errors = np.array(errors)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))
        return {'rmse': round(rmse, 4), 'mae': round(mae, 4)}

    def precision_recall_at_k(
        self,
        test_ratings: List[Dict],
        k: int = 5,
        threshold: float = 3.5
    ) -> Dict[str, float]:
        """
        Compute Precision@K and Recall@K across all users in the test set.

        Args:
            test_ratings: List of {'user_id', 'item_id', 'rating'}
            k:            Cut-off rank.
            threshold:    Minimum rating to consider "relevant".

        Returns:
            {'precision_at_k': …, 'recall_at_k': …, 'f1_at_k': …}
        """
        if not self.is_fitted or not test_ratings:
            return {'precision_at_k': 0.0, 'recall_at_k': 0.0, 'f1_at_k': 0.0}

        from collections import defaultdict
        user_test = defaultdict(list)
        for r in test_ratings:
            user_test[r['user_id']].append((r['item_id'], r['rating']))

        precisions, recalls = [], []

        for user_id, actual_items in user_test.items():
            if user_id not in self.user_ids:
                continue

            # Items rated at or above threshold in the test set
            relevant = {iid for iid, rating in actual_items if rating >= threshold}
            if not relevant:
                continue

            u_idx = self.user_ids.index(user_id)
            all_preds = [(self.item_ids[i], float(self.predicted_ratings[u_idx, i]))
                         for i in range(len(self.item_ids))]
            all_preds.sort(key=lambda x: x[1], reverse=True)

            top_k = {iid for iid, _ in all_preds[:k]}
            hit = len(relevant & top_k)
            precisions.append(hit / k)
            recalls.append(hit / len(relevant))

        p = float(np.mean(precisions)) if precisions else 0.0
        r = float(np.mean(recalls)) if recalls else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        return {
            'precision_at_k': round(p, 4),
            'recall_at_k':    round(r, 4),
            'f1_at_k':        round(f1, 4)
        }
