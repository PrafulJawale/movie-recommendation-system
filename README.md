# 🎬 CineMatch — AI-Based Movie Recommendation System

> **AI-Based Movie Recommendation System Using Content-Based Filtering**  
> Fundamentals of Artificial Intelligence (FAI) — Mini Project  
> Second-Year B.Tech, Artificial Intelligence and Machine Learning

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Objectives](#2-objectives)
3. [Proposed Solution](#3-proposed-solution)
4. [Introduction to Content-Based Filtering](#4-introduction-to-content-based-filtering)
5. [TF-IDF Explained](#5-tf-idf-explained)
6. [Cosine Similarity Explained](#6-cosine-similarity-explained)
7. [System Workflow](#7-system-workflow)
8. [Technology Stack](#8-technology-stack)
9. [Dataset](#9-dataset)
10. [Project Structure](#10-project-structure)
11. [Installation (Windows)](#11-installation-windows)
12. [Running the Application](#12-running-the-application)
13. [Demonstrating the Project](#13-demonstrating-the-project)
14. [Testing](#14-testing)
15. [Limitations](#15-limitations)
16. [Future Scope](#16-future-scope)
17. [Viva Preparation](#17-viva-preparation)

---

## 1. Problem Statement

With thousands of movies available on streaming platforms, users often struggle to discover films they will enjoy. A movie recommendation system addresses this by automatically suggesting movies that are similar to ones the user already likes.

This project builds such a system using **content-based filtering**, which recommends movies based on their descriptive features — genres, plot overview, and thematic keywords — rather than requiring user ratings or a large user base.

---

## 2. Objectives

- Build an interactive, AI-powered movie recommendation application.
- Implement genuine content-based filtering using TF-IDF vectorisation and cosine similarity.
- Provide ranked, explainable recommendations computed dynamically from movie metadata.
- Design a professional, accessible interface using Streamlit.
- Document the system clearly for academic submission and demonstration.

---

## 3. Proposed Solution

CineMatch is a **Streamlit web application** that:

1. Loads a curated dataset of 100 movies with genres, overviews, and keywords.
2. Converts each movie's textual features into a numerical TF-IDF vector.
3. When a user selects a movie, computes cosine similarity between that movie's vector and all other movies.
4. Returns the top-N most similar movies, ranked by similarity score.
5. Explains each recommendation using shared genres and thematic terms.

No API key, database, neural network, or internet connection is required after the initial package installation.

---

## 4. Introduction to Content-Based Filtering

**Content-based filtering** is a recommendation strategy that analyses the *properties of items* to find similar items. It does **not** require any user ratings, viewing history, or demographic data.

### How it differs from collaborative filtering

| Aspect | Content-Based (this project) | Collaborative Filtering |
|--------|------------------------------|------------------------|
| Data used | Movie metadata (genres, overview) | User ratings & behaviour |
| User data needed | No | Yes (many users) |
| Cold-start problem | No — works for new users | Yes — needs history |
| Discovery | Limited to similar metadata | Can surface unexpected gems |
| Transparency | Easy to explain | Often a "black box" |

**CineMatch uses content-based filtering** because it is transparent, does not require any user data, and clearly demonstrates the underlying mathematics.

---

## 5. TF-IDF Explained

**TF-IDF** stands for **Term Frequency – Inverse Document Frequency**.

It converts text into numerical vectors by assigning each word a weight that reflects how important that word is to one particular document relative to all other documents.

### Term Frequency (TF)

How often a word appears in a single movie's feature string, normalised by the total number of words in that string.

```
TF(word, movie) = (count of word in movie) / (total words in movie)
```

### Inverse Document Frequency (IDF)

A penalty for words that appear in *many* movies (e.g., "the", "a"). Words rare across the collection receive a higher IDF and therefore more importance.

```
IDF(word) = log( total movies / number of movies containing word )
```

### TF-IDF weight

```
TF-IDF(word, movie) = TF(word, movie) × IDF(word)
```

### Intuition

- The word **"space"** in *Interstellar* receives a high score because it is specific and distinctive.
- The word **"the"** receives a low score because it appears in every movie.
- This means movies that share distinctive, rare words are scored as more similar.

In this project, scikit-learn's `TfidfVectorizer` handles this computation, with `sublinear_tf=True` (log-scale term frequency) and English stop-word removal.

---

## 6. Cosine Similarity Explained

Once each movie is represented as a TF-IDF vector, **cosine similarity** measures how similar two vectors are.

### Formula

$$\text{similarity}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|}$$

Where:
- $A \cdot B$ is the dot product of the two vectors.
- $\|A\|$ and $\|B\|$ are the magnitudes (lengths) of the vectors.

### Intuition

Cosine similarity measures the **angle** between two vectors in high-dimensional space, not their magnitude.

- A score of **1.0** means the two movies have identical feature directions (maximum similarity).
- A score of **0.0** means the movies share no common features.

### Why cosine and not Euclidean distance?

Documents of different lengths produce vectors of different magnitudes. Cosine similarity normalises for this, focusing on the *direction* (what themes are present) rather than the *size* (how long the description is).

---

## 7. System Workflow

```
User selects a movie
        │
        ▼
data_loader.py reads data/movies.csv
        │
        ▼
Validate schema → normalise text → handle missing values
        │
        ▼
recommender.py combines: genres (×2) + overview + keywords → "soup"
        │
        ▼
TfidfVectorizer.fit_transform(soups) → sparse TF-IDF matrix [N × vocab]
        │
        ▼
Compute cosine_similarity(query_row, full_matrix) → similarity scores
        │
        ▼
Sort by descending score → exclude selected movie → take top-N
        │
        ▼
Generate plain-English explanation from shared genres & keywords
        │
        ▼
Display ranked recommendation cards in Streamlit UI
```

---

## 8. Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.13 |
| Web interface | Streamlit | ≥ 1.32 |
| ML / NLP | scikit-learn | ≥ 1.4 |
| Data handling | Pandas | ≥ 2.1 |
| Numerical computing | NumPy | ≥ 1.26 |
| Testing | Pytest | ≥ 8.0 |

---

## 9. Dataset

### Source

**Curated sample dataset** — 100 hand-selected films spanning 10+ genres.

This is *not* a download of a large public dataset like MovieLens or TMDB.  
It is a curated local CSV bundled directly with the project for offline use.

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `movie_id` | int | Unique numeric identifier |
| `title` | str | Movie title |
| `year` | int | Release year |
| `genres` | str | Pipe-delimited genres (e.g., `"Action\|Drama"`) |
| `overview` | str | Short plot description |
| `keywords` | str | Space-separated thematic tags |

### Genre Coverage

Action · Adventure · Animation · Biography · Comedy · Crime · Drama ·
Fantasy · History · Horror · Music · Mystery · Romance · Science Fiction ·
Thriller · War

### Limitations

- 100 movies: recommendations are less nuanced than on a dataset of thousands.
- No user ratings, no box-office data, no cast information.
- No poster images are bundled (no API key or third-party service required).

---

## 10. Project Structure

```
CineMatch_FAI_Mini_Project/
│
├── app.py              # Streamlit interface and application flow
├── recommender.py      # TF-IDF, cosine similarity, ranking, explanations
├── data_loader.py      # Dataset loading, validation, normalisation
│
├── data/
│   └── movies.csv      # 100-movie curated offline dataset
│
├── tests/
│   ├── __init__.py
│   └── test_recommender.py   # 30 automated pytest tests
│
├── assets/
│   └── placeholder.svg       # Poster placeholder artwork
│
├── requirements.txt    # Python dependencies
├── .gitignore          # Ignored files
└── README.md           # This document
```

---

## 11. Installation (Windows)

Open **VS Code Terminal** (Ctrl + `` ` ``) and run:

```powershell
# 1. Navigate to the project folder (adjust path if needed)
cd "C:\Users\YourName\Downloads\CineMatch_FAI_Mini_Project"

# 2. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## 12. Running the Application

```powershell
# From inside the CineMatch_FAI_Mini_Project folder
streamlit run app.py
```

Streamlit will open automatically in your default browser at:

```
http://localhost:8501
```

If the browser does not open, navigate to that URL manually.

To stop the application: press `Ctrl + C` in the terminal.

---

## 13. Demonstrating the Project

### Suggested demonstration flow

1. **Open the app** — `streamlit run app.py`
2. **Browse the dataset** — click the **Browse Movies** tab to show the 100 movies.
3. **Select a movie** — return to **Get Recommendations**, choose *Inception* from the dropdown.
4. **Set parameters** — use the sidebar to set 5 recommendations.
5. **Click "Find Similar Movies"** — point out the similarity scores and genre pills.
6. **Explain a result** — hover over the explanation text below a card. Note shared genres like *Action | Thriller*.
7. **Change the movie** — select *The Notebook* (Romance/Drama) and regenerate. Show how the results change completely.
8. **Explain the algorithm** — click the **How It Works** tab and walk through the 6 steps.
9. **Filter by genre** — use the sidebar genre dropdown. Select *Animation* and show only animated films.
10. **About page** — show the **About** tab: technology stack, dataset details, and limitations.

### Key demonstration points

- Results change when you select a different movie — they are **not hardcoded**.
- Similarity scores are numeric values from the algorithm, **not training accuracy**.
- The selected movie **never appears** in its own recommendations.
- A sci-fi movie returns mostly sci-fi recommendations; a romance returns mostly romance.

---

## 14. Testing

### Running tests

```powershell
# From the project root
python -m pytest tests/ -v
```

### Test categories

| Category | Tests |
|----------|-------|
| Dataset loading | 8 tests |
| Engine correctness | 9 tests |
| Edge cases & robustness | 12 tests |
| **Total** | **30 tests** |

### What is tested

1. Bundled dataset loads successfully.
2. Required fields are validated.
3. Missing optional text fields do not crash the engine.
4. Known movies produce valid recommendations.
5. Selected movie is excluded from its own results.
6. Recommendation counts are respected.
7. Unknown movie IDs raise clear errors.
8. Empty and malformed datasets produce a clear DataLoadError.
9. Duplicate records are handled (keep first, warn).
10. Similarity scores are finite floats in [0, 1].
11. Movies with empty descriptions or missing genres are handled safely.
12. Results are deterministic on the same data.

> ⚠️ **Actual test results:** 30 passed, 0 failed (Python 3.13, pandas 3.0, scikit-learn 1.8).

---

## 15. Limitations

1. **Small dataset:** With only 100 movies, recommendations lack the nuance of production systems.
2. **Metadata-only:** Recommendations depend entirely on written descriptions and genre labels. Quality varies.
3. **No personalisation:** There are no user profiles, no ratings, no viewing history. The same movie always returns the same recommendations regardless of who asks.
4. **No collaborative signals:** A movie beloved by users with similar taste cannot surface unless its text is also similar.
5. **Language dependency:** TF-IDF works on English text. Non-English or poorly described movies may mismatch.
6. **No poster images:** No image API is used, so poster thumbnails are not displayed.
7. **Similarity ≠ quality:** A high similarity score does not mean the recommended film is better, more popular, or more likely to be enjoyed.

---

## 16. Future Scope

| Improvement | Description |
|-------------|-------------|
| Larger dataset | Integrate MovieLens 25M or TMDB dataset (hundreds of thousands of movies) |
| Collaborative filtering | Add user ratings; combine with content-based in a hybrid model |
| Poster images | Integrate TMDB API (free key) for movie poster thumbnails |
| Cast & crew features | Include director, cast names as additional features |
| Deep learning | Sentence-BERT embeddings for richer semantic similarity |
| Deployment | Streamlit Community Cloud, Docker, or Hugging Face Spaces |
| User profiles | Session-based preference tracking for personalised results |
| Evaluation metrics | Precision@K, Recall@K with a held-out test split |

---

## 17. Viva Preparation

### Q1: What problem does this application solve?
**A:** Users struggle to find good movies among thousands of options. CineMatch solves this by automatically recommending movies similar to one the user already likes, using their descriptive metadata — genres, plot overview, and keywords.

---

### Q2: What is content-based filtering?
**A:** Content-based filtering is a recommendation technique that compares the *properties of items*, not the behaviour of users. A movie is recommended because its description or genres are similar to those of a movie the user selected. It requires no user ratings or browsing history.

---

### Q3: What is TF-IDF and why is it used here?
**A:** TF-IDF stands for Term Frequency–Inverse Document Frequency. It converts text into numbers. A word that appears often in one movie's description but rarely across all movies gets a high TF-IDF score — it is distinctive. Words like "the" appear everywhere and get low scores. We use it because it captures how *meaningful* each word is relative to the whole collection, not just how frequently it appears.

---

### Q4: What is cosine similarity?
**A:** Cosine similarity measures the angle between two vectors in a high-dimensional space. It returns a value between 0 and 1. A score of 1 means the two movies have identical feature vectors (same genres, same themes); 0 means no shared features. We use cosine similarity because it normalises for description length, focusing on *what* features are present rather than *how much* text there is.

**Formula:**
> similarity(A, B) = (A · B) / (||A|| × ||B||)

---

### Q5: Why is the selected movie excluded from its own recommendations?
**A:** A movie is perfectly identical to itself, so it would always score 1.0 and appear first. That is useless to the user. We explicitly skip the query movie's index when building the ranked list.

---

### Q6: What data does the system use for each movie?
**A:** Each movie record contains: a unique ID, title, release year, genres (pipe-separated), a plot overview, and optional keywords. The genres are weighted by repeating them twice in the feature string to give them extra importance in the similarity calculation.

---

### Q7: How are the recommendations ranked?
**A:** All movies are scored by cosine similarity against the selected movie's TF-IDF vector. They are sorted in descending order — highest similarity first. The top-N are returned. Ties in scores are broken by original dataset order, which is consistent across runs (deterministic).

---

### Q8: What is the difference between similarity and accuracy?
**A:** *Similarity* measures how closely two movies' textual features match. *Accuracy* is a training metric that measures how often a model's predictions match true labels, and it requires a labelled test set. CineMatch does not train a model and has no accuracy metric. Calling similarity "accuracy" would be misleading.

---

### Q9: What are the main limitations of this system?
**A:** The dataset has only 100 movies. There is no personalisation — everyone gets the same recommendations for the same movie. The system uses no collaborative signals, ratings, or user data. Recommendations depend entirely on textual description quality. No neural network was trained.

---

### Q10: How could you improve this project?
**A:** Use a larger dataset (MovieLens or TMDB with thousands of movies); add collaborative filtering based on user ratings; use BERT-based sentence embeddings for deeper semantic similarity; integrate TMDB's API for poster images; deploy online; add evaluation metrics like Precision@K.

---

### Q11: Is this a machine learning or deep learning project?
**A:** It is a **classical machine learning / NLP** project. TF-IDF is a statistical text representation technique. Cosine similarity is a linear algebra operation. No neural network is trained. This is intentional — the system is designed to be transparent, explainable, and easy to understand at the B.Tech level.

---

### Q12: If two movies have the same genre but very different plots, what happens?
**A:** They will have some similarity (from shared genre tokens, which are weighted by being repeated), but the overview and keyword terms will lower the overall score if the plots differ significantly. The system finds a balance between genre-based and description-based similarity.

---

*Built with Python, scikit-learn, Streamlit, and Pandas.*
