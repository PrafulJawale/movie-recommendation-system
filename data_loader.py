"""
data_loader.py — CineMatch Dataset Loading and Validation

Responsible for loading movies.csv, validating its schema,
normalising missing fields, and providing a safe fallback if needed.
"""

import os
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Required columns every valid dataset must contain ─────────────────────────
REQUIRED_COLUMNS: list[str] = ["movie_id", "title", "year", "genres", "overview"]

# ── Optional columns — used when present, safely ignored when absent ──────────
OPTIONAL_COLUMNS: list[str] = ["keywords"]

# ── Path resolution ───────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent
DEFAULT_DATA_PATH = _PROJECT_ROOT / "data" / "movies.csv"


class DataLoadError(Exception):
    """Raised when the dataset cannot be loaded or is fundamentally unusable."""


def load_movies(path: str | os.PathLike | None = None) -> pd.DataFrame:
    """
    Load and validate the movie dataset.

    Parameters
    ----------
    path : path-like, optional
        Path to a CSV file.  Defaults to data/movies.csv relative to this file.

    Returns
    -------
    pd.DataFrame
        Cleaned, validated DataFrame ready for the recommendation engine.

    Raises
    ------
    DataLoadError
        If the file is missing, empty, or lacks required columns.
    """
    csv_path = Path(path) if path is not None else DEFAULT_DATA_PATH

    # ── 1. File existence check ────────────────────────────────────────────────
    if not csv_path.exists():
        raise DataLoadError(
            f"Dataset file not found: {csv_path}\n"
            "Make sure data/movies.csv is present in the project directory."
        )

    # ── 2. Load CSV ────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as exc:
        raise DataLoadError(f"Failed to parse CSV file '{csv_path}': {exc}") from exc

    # ── 3. Empty dataset guard ─────────────────────────────────────────────────
    if df.empty:
        raise DataLoadError(f"Dataset file '{csv_path}' is empty.")

    # ── 4. Required-column validation ─────────────────────────────────────────
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DataLoadError(
            f"Dataset is missing required column(s): {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # ── 5. Normalise types ─────────────────────────────────────────────────────
    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # ── 6. Normalise text fields — replace NaN / 'nan' string with empty str ──
    text_cols = ["title", "genres", "overview"] + [
        c for c in OPTIONAL_COLUMNS if c in df.columns
    ]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()
        # Guard against literal "nan" strings produced by some CSV writers
        df[col] = df[col].replace("nan", "")

    # ── 7. Drop rows whose movie_id is not a valid number ─────────────────────
    invalid_ids = df["movie_id"].isna()
    if invalid_ids.any():
        logger.warning(
            "%d row(s) with invalid movie_id dropped.", invalid_ids.sum()
        )
        df = df[~invalid_ids].copy()

    if df.empty:
        raise DataLoadError("No valid rows remain after cleaning.")

    # ── 8. Duplicate movie_id detection ───────────────────────────────────────
    dup_mask = df.duplicated(subset="movie_id", keep="first")
    if dup_mask.any():
        logger.warning(
            "%d duplicate movie_id(s) found; keeping first occurrence.",
            dup_mask.sum(),
        )
        df = df[~dup_mask].copy()

    # ── 9. Reset index for stable integer-based lookups ───────────────────────
    df = df.reset_index(drop=True)

    # ── 10. Ensure optional columns exist (empty string) ─────────────────────
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    logger.info(
        "Dataset loaded successfully: %d movies from '%s'.", len(df), csv_path
    )
    return df


def get_genre_list(df: pd.DataFrame) -> list[str]:
    """
    Extract a sorted, unique list of all genres present in the dataset.

    The 'genres' column is expected to use '|' as a delimiter, e.g.
    "Action|Adventure|Comedy".
    """
    all_genres: set[str] = set()
    for entry in df["genres"].dropna():
        for genre in str(entry).split("|"):
            genre = genre.strip()
            if genre:
                all_genres.add(genre)
    return sorted(all_genres)


def filter_by_genre(df: pd.DataFrame, genre: str) -> pd.DataFrame:
    """Return only movies that include *genre* (case-insensitive)."""
    if not genre or genre.lower() == "all":
        return df
    mask = df["genres"].str.contains(genre, case=False, na=False)
    return df[mask].copy()
