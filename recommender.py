"""
recommender.py — CineMatch Recommendation Engine

Implements content-based filtering using TF-IDF vectorisation and
cosine similarity.  This module is intentionally independent of
Streamlit so it can be tested and reused freely.

Algorithm summary
-----------------
1. Combine each movie's textual fields (genres, overview, keywords)
   into one 'soup' string.
2. Convert all soups into a sparse TF-IDF matrix.
3. For a query movie, compute cosine similarity against every other
   movie's TF-IDF vector.
4. Rank candidates by descending similarity, exclude the query movie,
   and return the top-N results.

Cosine similarity:
    similarity(A, B) = (A · B) / (||A|| × ||B||)

A score of 1 means identical feature vectors; 0 means no shared features.
"""

from __future__ import annotations

import logging
import re
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

logger = logging.getLogger(__name__)


# ── Public result type ────────────────────────────────────────────────────────

class Recommendation(NamedTuple):
    """A single recommendation result."""

    movie_id: int
    title: str
    year: int | None
    genres: str
    overview: str
    similarity_score: float   # cosine similarity in [0, 1]
    explanation: str          # human-readable reason for recommendation


# ── Engine ────────────────────────────────────────────────────────────────────

class RecommendationEngine:
    """
    Content-based movie recommendation engine.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame produced by data_loader.load_movies().
    max_features : int
        Maximum vocabulary size for TF-IDF (default 10 000 is ample for
        small datasets and keeps memory reasonable for large ones).
    """

    def __init__(self, df: pd.DataFrame, max_features: int = 10_000) -> None:
        if df is None or df.empty:
            raise ValueError("Cannot build a recommendation engine from an empty dataset.")

        self._df = df.reset_index(drop=True).copy()
        self._max_features = max_features

        # Build and cache the TF-IDF matrix at construction time.
        self._tfidf_matrix, self._vectorizer = self._build_tfidf()

        logger.info(
            "RecommendationEngine ready: %d movies, TF-IDF vocab size %d.",
            len(self._df),
            len(self._vectorizer.vocabulary_),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def recommend(
        self,
        movie_id: int,
        n: int = 5,
    ) -> list[Recommendation]:
        """
        Return up to *n* recommendations for the movie identified by *movie_id*.

        Parameters
        ----------
        movie_id : int
            Stable identifier from the dataset's 'movie_id' column.
        n : int
            Number of recommendations requested (clamped to valid range).

        Returns
        -------
        list[Recommendation]
            Ranked by descending cosine similarity.  May be shorter than *n*
            if not enough candidate movies exist.

        Raises
        ------
        ValueError
            If the movie_id is not found in the dataset.
        """
        n = max(1, min(n, len(self._df) - 1))  # clamp safely

        # ── Locate the query movie ─────────────────────────────────────────────
        matches = self._df.index[self._df["movie_id"] == movie_id].tolist()
        if not matches:
            raise ValueError(
                f"movie_id {movie_id!r} not found in the dataset. "
                "It may have been removed during cleaning."
            )
        query_idx = matches[0]

        # ── Compute cosine similarities on-the-fly (row vs. matrix) ───────────
        # linear_kernel on a single row is equivalent to cosine similarity when
        # using TfidfVectorizer (which already L2-normalises its output).
        query_vec = self._tfidf_matrix[query_idx]
        sim_scores = linear_kernel(query_vec, self._tfidf_matrix).flatten()

        # ── Rank all other movies ──────────────────────────────────────────────
        # Create (index, score) pairs, exclude the query movie itself.
        ranked = [
            (idx, float(score))
            for idx, score in enumerate(sim_scores)
            if idx != query_idx
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)

        query_row = self._df.iloc[query_idx]
        results: list[Recommendation] = []

        for idx, score in ranked[:n]:
            row = self._df.iloc[idx]
            results.append(
                Recommendation(
                    movie_id=int(row["movie_id"]),
                    title=str(row["title"]),
                    year=int(row["year"]) if pd.notna(row.get("year")) else None,
                    genres=str(row.get("genres", "")),
                    overview=str(row.get("overview", "")),
                    similarity_score=score,
                    explanation=self._explain(query_row, row, score),
                )
            )

        return results

    def get_movie_by_id(self, movie_id: int) -> pd.Series | None:
        """Return the DataFrame row for *movie_id*, or None if not found."""
        matches = self._df[self._df["movie_id"] == movie_id]
        if matches.empty:
            return None
        return matches.iloc[0]

    @property
    def movies(self) -> pd.DataFrame:
        """Read-only view of the internal movie DataFrame."""
        return self._df.copy()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_tfidf(self) -> tuple:
        """
        Construct the TF-IDF matrix from combined feature strings.

        The 'soup' for each movie combines genres, overview, and keywords.
        Genres receive extra weight by being repeated — a simple but effective
        technique that amplifies genre similarity without modifying the core
        TF-IDF mathematics.
        """
        soups = self._df.apply(self._make_soup, axis=1)

        vectorizer = TfidfVectorizer(
            max_features=self._max_features,
            stop_words="english",
            ngram_range=(1, 2),   # unigrams + bigrams capture phrases
            min_df=1,
            sublinear_tf=True,    # log-scale term frequency
        )
        matrix = vectorizer.fit_transform(soups)
        return matrix, vectorizer

    @staticmethod
    def _make_soup(row: pd.Series) -> str:
        """
        Combine textual features into a single string for TF-IDF.

        Genres are repeated twice to give them extra weight in the
        similarity computation.  The pipe delimiter in genres is replaced
        with spaces so each genre becomes a separate token.
        """
        genres_raw = str(row.get("genres", ""))
        genres_clean = re.sub(r"[|]+", " ", genres_raw).strip()
        # Repeat genres to boost their importance
        genres_weighted = (genres_clean + " ") * 2

        overview = str(row.get("overview", "")).strip()
        keywords = str(row.get("keywords", "")).strip()

        return f"{genres_weighted}{overview} {keywords}".strip()

    @staticmethod
    def _explain(
        query_row: pd.Series,
        rec_row: pd.Series,
        score: float,
    ) -> str:
        """
        Produce a deterministic, data-driven explanation for a recommendation.

        Logic:
        1. Find genres shared between the query and the recommendation.
        2. Find keywords that appear in both overviews / keyword fields.
        3. Compose a short, human-readable sentence from these facts.
        """
        query_genres = set(
            g.strip() for g in str(query_row.get("genres", "")).split("|") if g.strip()
        )
        rec_genres = set(
            g.strip() for g in str(rec_row.get("genres", "")).split("|") if g.strip()
        )
        shared_genres = sorted(query_genres & rec_genres)

        # Simple keyword overlap: tokenise both keyword/overview fields
        def _tokens(row: pd.Series) -> set[str]:
            text = (
                str(row.get("keywords", "")) + " " + str(row.get("overview", ""))
            ).lower()
            # Keep only alphabetic tokens longer than 3 characters
            return {t for t in re.findall(r"[a-z]{4,}", text) if t not in _STOPWORDS}

        shared_keywords = sorted(_tokens(query_row) & _tokens(rec_row))[:5]

        parts: list[str] = []
        if shared_genres:
            genre_str = ", ".join(shared_genres)
            parts.append(f"Shares genres: **{genre_str}**")
        if shared_keywords:
            kw_str = ", ".join(shared_keywords)
            parts.append(f"Common themes: *{kw_str}*")

        if not parts:
            parts.append("Matched on combined textual features")

        score_pct = f"{score * 100:.1f}%"
        return f"{' · '.join(parts)} (similarity: {score_pct})"


# ── Stop-word set for explanation generation ──────────────────────────────────
_STOPWORDS: frozenset[str] = frozenset(
    """
    this that with from have been into when then their they them also
    while about after which would could should very more most some
    such only even over into upon himself herself itself himself
    himself what where both after during before through across against
    where while their there these those being find finds
    """.split()
)
