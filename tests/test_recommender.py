"""
tests/test_recommender.py

Deterministic pytest test suite for the CineMatch recommendation engine.
All tests are offline — no network access required.
"""

import math
import pytest
import pandas as pd

from data_loader import load_movies, DataLoadError, get_genre_list, filter_by_genre
from recommender import RecommendationEngine


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_df():
    """Load the actual bundled dataset once for the whole test module."""
    return load_movies()


@pytest.fixture(scope="module")
def real_engine(real_df):
    """Build the engine once from the bundled dataset."""
    return RecommendationEngine(real_df)


@pytest.fixture
def minimal_df():
    """A tiny in-memory DataFrame that covers edge cases."""
    return pd.DataFrame(
        {
            "movie_id": [1, 2, 3, 4, 5],
            "title": [
                "Space Odyssey",
                "Galaxy Warriors",
                "Love in Paris",
                "Comedy Capers",
                "Space Explorer",
            ],
            "year": [2000, 2001, 2002, 2003, 2004],
            "genres": [
                "Science Fiction|Adventure",
                "Science Fiction|Action",
                "Romance|Drama",
                "Comedy",
                "Science Fiction|Drama",
            ],
            "overview": [
                "Astronauts explore deep space searching for alien life forms beyond known galaxies.",
                "Warriors fight alien invaders across the galaxy to defend humanity.",
                "Two strangers fall in love in the romantic streets of Paris.",
                "A group of comedians cause hilarious chaos at a country fair.",
                "A lone explorer ventures into uncharted space territories.",
            ],
            "keywords": [
                "space aliens exploration",
                "space aliens fight",
                "love romance Paris",
                "comedy funny slapstick",
                "space exploration adventure",
            ],
        }
    )


@pytest.fixture
def minimal_engine(minimal_df):
    return RecommendationEngine(minimal_df)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoader:
    def test_bundled_dataset_loads(self, real_df):
        """The bundled dataset must load without error."""
        assert isinstance(real_df, pd.DataFrame)
        assert not real_df.empty

    def test_required_columns_present(self, real_df):
        """All required columns must be present in the loaded DataFrame."""
        for col in ("movie_id", "title", "year", "genres", "overview"):
            assert col in real_df.columns, f"Missing required column: {col}"

    def test_minimum_row_count(self, real_df):
        """Bundled dataset should have at least 50 movies."""
        assert len(real_df) >= 50

    def test_no_duplicate_movie_ids(self, real_df):
        """movie_id column must be unique after loading."""
        assert real_df["movie_id"].duplicated().sum() == 0

    def test_text_fields_are_strings(self, real_df):
        """Text columns must contain string data with no raw NaN values.
        
        Accepts both legacy object dtype (pandas < 3.0) and the new
        StringDtype introduced in pandas 3.0.
        """
        for col in ("title", "genres", "overview"):
            # pandas 3.0 returns StringDtype; older versions return object
            dtype_name = str(real_df[col].dtype)
            assert dtype_name in ("object", "str") or dtype_name.startswith("string"), (
                f"Column '{col}' has unexpected dtype: {real_df[col].dtype}"
            )
            assert real_df[col].isna().sum() == 0

    def test_missing_file_raises_error(self, tmp_path):
        """Loading a non-existent file must raise DataLoadError."""
        with pytest.raises(DataLoadError, match="not found"):
            load_movies(tmp_path / "nonexistent.csv")

    def test_empty_csv_raises_error(self, tmp_path):
        """An empty CSV file must raise DataLoadError."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")
        with pytest.raises(DataLoadError):
            load_movies(empty_csv)

    def test_missing_required_column_raises_error(self, tmp_path):
        """A CSV missing 'overview' must raise DataLoadError."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("movie_id,title,year,genres\n1,Test,2000,Action\n")
        with pytest.raises(DataLoadError, match="overview"):
            load_movies(bad_csv)

    def test_get_genre_list(self, real_df):
        """Genre list must be a non-empty sorted list of strings."""
        genres = get_genre_list(real_df)
        assert isinstance(genres, list)
        assert len(genres) > 0
        assert genres == sorted(genres)

    def test_filter_by_genre_returns_subset(self, real_df):
        """Genre filter must return only movies containing that genre."""
        filtered = filter_by_genre(real_df, "Action")
        assert not filtered.empty
        assert all("Action" in g for g in filtered["genres"])

    def test_filter_all_returns_full_df(self, real_df):
        """Filtering with 'all' or empty string returns the full DataFrame."""
        assert len(filter_by_genre(real_df, "all")) == len(real_df)
        assert len(filter_by_genre(real_df, "")) == len(real_df)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Recommendation engine — core correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestRecommendationEngine:
    def test_engine_builds_from_real_data(self, real_engine):
        """Engine must build successfully from the bundled dataset."""
        assert real_engine is not None

    def test_empty_dataframe_raises(self):
        """Building an engine from an empty DataFrame must raise ValueError."""
        with pytest.raises(ValueError):
            RecommendationEngine(pd.DataFrame())

    def test_unknown_movie_id_raises(self, real_engine):
        """Requesting recommendations for an unknown ID must raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            real_engine.recommend(movie_id=999_999, n=5)

    def test_selected_movie_excluded_from_results(self, real_engine, real_df):
        """The query movie must never appear in its own recommendations."""
        for movie_id in real_df["movie_id"].head(10):
            recs = real_engine.recommend(int(movie_id), n=10)
            rec_ids = [r.movie_id for r in recs]
            assert int(movie_id) not in rec_ids, (
                f"movie_id {movie_id} appeared in its own recommendations."
            )

    def test_recommendation_count_respected(self, real_engine):
        """Engine must return at most the requested number of recommendations."""
        for n in (1, 3, 5, 10):
            recs = real_engine.recommend(1, n=n)
            assert len(recs) <= n

    def test_minimum_one_recommendation(self, real_engine):
        """Engine must return at least one recommendation for a movie with peers."""
        recs = real_engine.recommend(1, n=1)
        assert len(recs) >= 1

    def test_similarity_scores_are_valid(self, real_engine, real_df):
        """All similarity scores must be finite floats in [0, 1]."""
        for movie_id in real_df["movie_id"].head(20):
            recs = real_engine.recommend(int(movie_id), n=5)
            for rec in recs:
                assert isinstance(rec.similarity_score, float)
                assert math.isfinite(rec.similarity_score)
                # Allow a tiny float tolerance above 1 due to IEEE 754
                assert -1e-9 <= rec.similarity_score <= 1 + 1e-9

    def test_results_are_descending_by_score(self, real_engine, real_df):
        """Results must be ranked in descending similarity order."""
        for movie_id in real_df["movie_id"].head(10):
            recs = real_engine.recommend(int(movie_id), n=10)
            scores = [r.similarity_score for r in recs]
            assert scores == sorted(scores, reverse=True)

    def test_determinism(self, real_df):
        """Same input must always produce the same results."""
        eng1 = RecommendationEngine(real_df.copy())
        eng2 = RecommendationEngine(real_df.copy())
        recs1 = eng1.recommend(1, n=5)
        recs2 = eng2.recommend(1, n=5)
        assert [r.movie_id for r in recs1] == [r.movie_id for r in recs2]
        assert [r.similarity_score for r in recs1] == [r.similarity_score for r in recs2]

    def test_no_hardcoded_results(self, real_engine):
        """Verify recommendations differ meaningfully between unrelated movies."""
        sci_fi_recs = {r.movie_id for r in real_engine.recommend(2, n=10)}   # Inception
        romance_recs = {r.movie_id for r in real_engine.recommend(15, n=10)} # Titanic
        # The top results should NOT be identical for very different movies
        assert sci_fi_recs != romance_recs


# ─────────────────────────────────────────────────────────────────────────────
# 3. Edge cases and robustness
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_movie_with_empty_overview(self):
        """A movie with no overview must not crash the engine."""
        df = pd.DataFrame(
            {
                "movie_id": [1, 2, 3],
                "title": ["No Overview Film", "Normal Film A", "Normal Film B"],
                "year": [2000, 2001, 2002],
                "genres": ["Drama", "Drama|Action", "Comedy"],
                "overview": ["", "A hero saves the day in a dramatic showdown.", "Funny comedy chaos."],
                "keywords": ["", "hero action drama", "comedy funny"],
            }
        )
        engine = RecommendationEngine(df)
        recs = engine.recommend(1, n=2)
        # Should return something (even if score is low)
        assert isinstance(recs, list)

    def test_movie_with_empty_genres(self):
        """A movie with no genres must not crash the engine."""
        df = pd.DataFrame(
            {
                "movie_id": [1, 2, 3],
                "title": ["No Genre Film", "Action Movie A", "Action Movie B"],
                "year": [2000, 2001, 2002],
                "genres": ["", "Action|Thriller", "Action|Drama"],
                "overview": [
                    "A film about nothing in particular, really.",
                    "A thrilling action adventure with explosions.",
                    "Drama meets action in this epic blockbuster.",
                ],
                "keywords": ["", "action thriller", "action drama"],
            }
        )
        engine = RecommendationEngine(df)
        recs = engine.recommend(1, n=2)
        assert isinstance(recs, list)

    def test_duplicate_titles_handled(self, real_df):
        """Engine must still work even when two movies share a title (different IDs)."""
        dup_df = real_df.copy()
        # Artificially add a duplicate title with a unique ID
        extra = real_df.iloc[0].copy()
        extra["movie_id"] = 9999
        extra_df = pd.concat([dup_df, extra.to_frame().T], ignore_index=True)
        extra_df["movie_id"] = pd.to_numeric(extra_df["movie_id"])
        engine = RecommendationEngine(extra_df)
        recs = engine.recommend(9999, n=5)
        assert all(r.movie_id != 9999 for r in recs)

    def test_n_clamped_above_dataset_size(self, minimal_engine):
        """Requesting more recs than available movies must not crash."""
        recs = minimal_engine.recommend(1, n=1000)
        # At most len(df) - 1 results
        assert len(recs) <= 4

    def test_n_zero_clamped_to_one(self, minimal_engine):
        """n=0 must be clamped to 1."""
        recs = minimal_engine.recommend(1, n=0)
        assert len(recs) >= 1

    def test_get_movie_by_id_returns_correct_row(self, real_engine, real_df):
        """get_movie_by_id must return the correct row for a valid ID."""
        row = real_engine.get_movie_by_id(1)
        assert row is not None
        assert row["movie_id"] == 1

    def test_get_movie_by_id_returns_none_for_unknown(self, real_engine):
        """get_movie_by_id must return None for an unknown ID."""
        assert real_engine.get_movie_by_id(999_999) is None

    def test_recommendation_explanations_are_strings(self, real_engine, real_df):
        """Every recommendation must include a non-empty explanation string."""
        recs = real_engine.recommend(1, n=5)
        for rec in recs:
            assert isinstance(rec.explanation, str)
            assert len(rec.explanation) > 0

    def test_genre_similarity_makes_sense(self, minimal_engine):
        """Sci-fi movies should rank above a comedy for a sci-fi query."""
        recs = minimal_engine.recommend(1, n=4)  # Space Odyssey → Sci-Fi
        sci_fi_ids = {2, 5}   # Galaxy Warriors and Space Explorer
        comedy_id = 4          # Comedy Capers
        # At least one sci-fi movie should appear before comedy
        rec_ids = [r.movie_id for r in recs]
        if comedy_id in rec_ids:
            comedy_pos = rec_ids.index(comedy_id)
            sci_fi_positions = [rec_ids.index(i) for i in sci_fi_ids if i in rec_ids]
            if sci_fi_positions:
                assert min(sci_fi_positions) < comedy_pos, (
                    "Sci-fi movies should rank higher than comedy for a sci-fi query."
                )
