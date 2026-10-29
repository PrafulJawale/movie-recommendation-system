"""
app.py — CineMatch Streamlit Application
Rebuilt with native Streamlit components for maximum rendering reliability.
Custom HTML is used only for the top-level header and structural accents
where Streamlit has no equivalent — never for data-bearing content.
"""

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from data_loader import DataLoadError, filter_by_genre, get_genre_list, load_movies
from recommender import RecommendationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Must be the very first Streamlit call ─────────────────────────────────────
st.set_page_config(
    page_title="CineMatch — AI Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — theming and layout only, no content-bearing classes ──────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* ── Colour tokens ── */
    :root {
        --bg:         #0a0a0f;
        --surface:    #13131c;
        --surface2:   #1a1a27;
        --border:     #252535;
        --border2:    #303045;
        --accent:     #e63946;
        --accent2:    #c1121f;
        --gold:       #ffd166;
        --text:       #eeeef5;
        --muted:      #7e7e9a;
        --muted2:     #4a4a65;
        --radius:     10px;
        --radius-lg:  16px;
    }

    /* ── App shell ── */
    .stApp                       { background-color: var(--bg); }
    .block-container             { padding: 0 2rem 3rem; max-width: 1280px; }
    [data-testid="stSidebar"]   { background-color: var(--surface) !important; border-right: 1px solid var(--border); }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    /* ── Tabs ── */
    [data-baseweb="tab-list"]         { background: var(--surface) !important; border-radius: var(--radius); border: 1px solid var(--border); gap: 4px; padding: 4px; }
    [data-baseweb="tab"]              { border-radius: 8px !important; color: var(--muted) !important; font-weight: 500; padding: 8px 18px !important; }
    [aria-selected="true"][data-baseweb="tab"] { background: var(--accent) !important; color: #fff !important; }
    [data-baseweb="tab-highlight"]    { display: none !important; }
    [data-baseweb="tab-border"]       { display: none !important; }

    /* ── Native widget cosmetics ── */
    [data-baseweb="select"] > div     { background-color: var(--surface2) !important; border-color: var(--border2) !important; color: var(--text) !important; border-radius: var(--radius) !important; }
    [data-baseweb="input"]            { background-color: var(--surface2) !important; border-color: var(--border2) !important; border-radius: var(--radius) !important; }
    input[type="text"]                { color: var(--text) !important; }
    [data-testid="stSlider"] label    { color: var(--muted) !important; font-size: .82rem; }

    /* ── Buttons ── */
    [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
        border: none !important;
        border-radius: var(--radius) !important;
        font-weight: 700 !important;
        letter-spacing: .5px;
        box-shadow: 0 4px 20px rgba(230,57,70,.35) !important;
    }
    [data-testid="baseButton-primary"]:hover {
        box-shadow: 0 6px 28px rgba(230,57,70,.55) !important;
        transform: translateY(-1px);
    }
    [data-testid="baseButton-secondary"] {
        background: var(--surface2) !important;
        border: 1px solid var(--border2) !important;
        border-radius: var(--radius) !important;
        color: var(--text) !important;
    }

    /* ── Streamlit containers — card look ── */
    [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] { gap: 0; }

    /* ── Metric overrides ── */
    [data-testid="stMetric"] { background: var(--surface2); border-radius: var(--radius); padding: .6rem .8rem; border: 1px solid var(--border); }
    [data-testid="stMetricLabel"]  { color: var(--muted) !important; font-size: .72rem !important; text-transform: uppercase; letter-spacing: 1px; }
    [data-testid="stMetricValue"]  { color: var(--text) !important; font-size: 1.1rem !important; font-weight: 700; }
    [data-testid="stMetricDelta"]  { display: none; }

    /* ── Dividers ── */
    hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

    /* ── Caption text ── */
    [data-testid="stCaptionContainer"] p { color: var(--muted) !important; font-size: .8rem; }

    /* ── Info / warning / error boxes ── */
    [data-testid="stAlert"] { border-radius: var(--radius) !important; border-left-width: 3px !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar              { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track        { background: var(--surface); }
    ::-webkit-scrollbar-thumb        { background: var(--border2); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover  { background: var(--accent); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Caching ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_data() -> pd.DataFrame:
    return load_movies()


@st.cache_resource(show_spinner=False)
def _build_engine(_df_key: int) -> RecommendationEngine:
    df = _load_data()
    return RecommendationEngine(df)


# ── Utility helpers ───────────────────────────────────────────────────────────

def _year_str(year) -> str:
    s = str(year)
    return s if s not in ("None", "<NA>", "nan", "") else "N/A"


def _fmt_genres(genres: str) -> str:
    """Return a readable comma-separated genre string."""
    parts = [g.strip() for g in str(genres).split("|") if g.strip()]
    return " · ".join(parts) if parts else "Unknown"


def _score_bar(score: float) -> str:
    """Return a simple Unicode bar representing similarity visually."""
    filled = round(score * 10)
    bar = "█" * filled + "░" * (10 - filled)
    return bar


# ── Card components built from native Streamlit widgets ──────────────────────

def _selected_card(title: str, year, genres: str, overview: str) -> None:
    """Render the 'your selection' card using only native Streamlit calls."""
    # Red left-border accent via a thin markdown line
    st.markdown(
        "<div style='height:3px;background:linear-gradient(90deg,#e63946,transparent);"
        "border-radius:3px;margin-bottom:12px'></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='font-size:.68rem;font-weight:700;letter-spacing:2px;"
        f"color:#e63946;text-transform:uppercase;margin:0 0 4px'>▶ YOUR SELECTION</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='font-size:1.35rem;font-weight:800;color:#eeeef5;margin:0 0 2px;line-height:1.3'>"
        f"{title}</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='font-size:.82rem;color:#7e7e9a;margin:0 0 10px'>"
        f"{_year_str(year)}  ·  {_fmt_genres(genres)}</p>",
        unsafe_allow_html=True,
    )
    if overview and overview not in ("nan", ""):
        st.markdown(
            f"<p style='font-size:.88rem;color:#b8b8d0;line-height:1.6;margin:0'>"
            f"{overview}</p>",
            unsafe_allow_html=True,
        )


def _rec_card(
    rank: int,
    title: str,
    year,
    genres: str,
    overview: str,
    score: float,
    explanation: str,
) -> None:
    """
    Render a single recommendation card using native Streamlit calls.
    All text content goes through Streamlit's own rendering path —
    no raw HTML is injected for data-bearing text.
    """
    # Top accent bar whose width encodes similarity
    bar_pct = max(8, int(score * 100 * 3.5))  # scale for visual effect, min 8%
    bar_pct = min(bar_pct, 100)
    st.markdown(
        f"<div style='height:2px;background:linear-gradient(90deg,#e63946 {bar_pct}%,"
        f"#252535 {bar_pct}%);border-radius:2px;margin-bottom:10px'></div>",
        unsafe_allow_html=True,
    )

    # Rank + title row
    c_rank, c_title = st.columns([1, 9])
    with c_rank:
        st.markdown(
            f"<p style='font-size:1.5rem;font-weight:900;color:#e63946;"
            f"opacity:.7;margin:0;line-height:1.2'>#{rank}</p>",
            unsafe_allow_html=True,
        )
    with c_title:
        st.markdown(
            f"<p style='font-size:1.05rem;font-weight:700;color:#eeeef5;"
            f"margin:0;line-height:1.3'>{title}</p>",
            unsafe_allow_html=True,
        )

    # Year · genres
    st.markdown(
        f"<p style='font-size:.78rem;color:#7e7e9a;margin:4px 0 8px;'>"
        f"{_year_str(year)} &nbsp;·&nbsp; {_fmt_genres(genres)}</p>",
        unsafe_allow_html=True,
    )

    # Similarity score badge
    pct = score * 100
    st.markdown(
        f"<span style='display:inline-block;background:#1e0509;"
        f"border:1px solid #5a1018;border-radius:20px;padding:2px 12px;"
        f"font-size:.72rem;color:#e63946;font-weight:700;margin-bottom:8px'>"
        f"⚡ {pct:.1f}% match &nbsp; {_score_bar(score)}</span>",
        unsafe_allow_html=True,
    )

    # Overview
    if overview and overview not in ("nan", ""):
        st.markdown(
            f"<p style='font-size:.83rem;color:#a0a0bc;line-height:1.6;margin:0 0 8px'>"
            f"{overview[:260]}{'…' if len(overview) > 260 else ''}</p>",
            unsafe_allow_html=True,
        )

    # Explanation
    if explanation:
        st.markdown(
            f"<p style='font-size:.75rem;color:#5566aa;border-top:1px solid #252535;"
            f"padding-top:7px;margin:0'>💡 {explanation}</p>",
            unsafe_allow_html=True,
        )


def _card_wrapper(content_fn, *args, selected=False, **kwargs):
    """Draw a visible card border around content using st.container."""
    border_color = "#e63946" if selected else "#252535"
    bg_color = "#160a0c" if selected else "#13131c"
    st.markdown(
        f"<div style='background:{bg_color};border:1px solid {border_color};"
        f"border-radius:10px;padding:16px 18px 14px;margin-bottom:14px'>",
        unsafe_allow_html=True,
    )
    content_fn(*args, **kwargs)
    st.markdown("</div>", unsafe_allow_html=True)


def _browse_card(title: str, year, genres: str, overview: str) -> None:
    st.markdown(
        f"<p style='font-size:.98rem;font-weight:700;color:#eeeef5;margin:0 0 2px'>{title}</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='font-size:.75rem;color:#7e7e9a;margin:0 0 7px'>"
        f"{_year_str(year)} &nbsp;·&nbsp; {_fmt_genres(genres)}</p>",
        unsafe_allow_html=True,
    )
    if overview and overview not in ("nan", ""):
        st.markdown(
            f"<p style='font-size:.8rem;color:#9090b0;line-height:1.55;margin:0'>"
            f"{overview[:200]}{'…' if len(overview) > 200 else ''}</p>",
            unsafe_allow_html=True,
        )


# ── Main app ─────────────────────────────────────────────────────────────────

def main() -> None:

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            padding: 2.5rem 1rem 2rem;
            text-align: center;
            background: linear-gradient(180deg, #110008 0%, #0a0a0f 100%);
            border-bottom: 1px solid #252535;
            margin: -1rem -2rem 2rem;
        ">
            <div style="font-size:3rem;font-weight:900;letter-spacing:5px;color:#fff;
                        text-shadow:0 0 40px rgba(230,57,70,.6);line-height:1">
                CINE<span style="color:#e63946">MATCH</span>
            </div>
            <div style="font-size:1.1rem;color:#7e7e9a;margin-top:.6rem;letter-spacing:.5px">
                Discover your next favourite film.
            </div>
            <div style="font-size:.82rem;color:#4a4a65;margin-top:.5rem">
                Content-based AI recommendations · TF-IDF &amp; Cosine Similarity · 100 curated films
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load data ──────────────────────────────────────────────────────────────
    with st.spinner("Loading movie dataset…"):
        try:
            df = _load_data()
            engine = _build_engine(hash(tuple(df["movie_id"].tolist())))
        except DataLoadError as exc:
            st.error(f"**Dataset error:** {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"**Unexpected error:** {exc}")
            st.stop()

    genre_list = ["All Genres"] + get_genre_list(df)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<h2 style='color:#e63946;font-size:1.4rem;letter-spacing:3px;"
            "font-weight:900;margin:0 0 1.5rem'>🎬 CineMatch</h2>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<p style='font-size:.7rem;font-weight:700;letter-spacing:2px;"
            "color:#7e7e9a;text-transform:uppercase;margin-bottom:.4rem'>Genre Filter</p>",
            unsafe_allow_html=True,
        )
        selected_genre = st.selectbox("Genre", genre_list, label_visibility="collapsed")

        st.divider()

        st.markdown(
            "<p style='font-size:.7rem;font-weight:700;letter-spacing:2px;"
            "color:#7e7e9a;text-transform:uppercase;margin-bottom:.4rem'>Recommendations</p>",
            unsafe_allow_html=True,
        )
        n_recs = st.slider("How many?", min_value=1, max_value=15, value=5)

        st.divider()

        # Dataset stats
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Movies", len(df))
        with col_s2:
            st.metric("Genres", len(genre_list) - 1)

        st.divider()
        st.markdown(
            "<p style='font-size:.72rem;color:#4a4a65;text-align:center;line-height:1.6'>"
            "FAI Mini Project<br/>B.Tech AI/ML · 2nd Year<br/>Content-Based Filtering</p>",
            unsafe_allow_html=True,
        )

    # ── Filtered view ──────────────────────────────────────────────────────────
    filtered_df = filter_by_genre(df, "" if selected_genre == "All Genres" else selected_genre)

    if filtered_df.empty:
        st.warning(f"No movies found for genre **{selected_genre}**.")
        st.stop()

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab_rec, tab_browse, tab_algo, tab_about = st.tabs(
        ["🎯  Recommendations", "🎬  Browse", "⚙️  How It Works", "ℹ️  About"]
    )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Recommendations
    # ══════════════════════════════════════════════════════════════════════════
    with tab_rec:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # ── Controls row ──────────────────────────────────────────────────────
        ctrl_left, ctrl_right = st.columns([3, 2], gap="large")

        with ctrl_left:
            st.markdown(
                "<p style='font-size:.72rem;font-weight:700;letter-spacing:2px;"
                "color:#7e7e9a;text-transform:uppercase;margin-bottom:.3rem'>Select a movie</p>",
                unsafe_allow_html=True,
            )
            movie_options = {
                row["title"]: int(row["movie_id"])
                for _, row in filtered_df.iterrows()
            }
            chosen_title = st.selectbox(
                "Movie",
                options=list(movie_options.keys()),
                label_visibility="collapsed",
            )
            chosen_id = movie_options[chosen_title]
            chosen_row = engine.get_movie_by_id(chosen_id)

            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            recommend_clicked = st.button(
                f"🎯  Find {n_recs} Similar Movies",
                type="primary",
                use_container_width=True,
            )

        with ctrl_right:
            st.markdown(
                "<p style='font-size:.72rem;font-weight:700;letter-spacing:2px;"
                "color:#7e7e9a;text-transform:uppercase;margin-bottom:.3rem'>Selected movie</p>",
                unsafe_allow_html=True,
            )
            if chosen_row is not None:
                # Inline card — uses only st.markdown with inline styles (no CSS classes)
                st.markdown(
                    f"<div style='background:#160a0c;border:1px solid #e63946;"
                    f"border-radius:10px;padding:16px 18px'>"
                    f"<p style='font-size:.62rem;font-weight:700;letter-spacing:2px;"
                    f"color:#e63946;text-transform:uppercase;margin:0 0 6px'>▶ Your Selection</p>"
                    f"<p style='font-size:1.15rem;font-weight:800;color:#eeeef5;"
                    f"margin:0 0 4px;line-height:1.3'>{chosen_row['title']}</p>"
                    f"<p style='font-size:.78rem;color:#7e7e9a;margin:0 0 10px'>"
                    f"{_year_str(chosen_row.get('year'))} &nbsp;·&nbsp; "
                    f"{_fmt_genres(str(chosen_row.get('genres', '')))}</p>"
                    f"<p style='font-size:.82rem;color:#b0b0cc;line-height:1.6;margin:0'>"
                    f"{str(chosen_row.get('overview',''))[:280]}"
                    f"{'…' if len(str(chosen_row.get('overview',''))) > 280 else ''}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── Session state ──────────────────────────────────────────────────────
        if "last_recs" not in st.session_state:
            st.session_state.last_recs = None
            st.session_state.last_movie_id = None

        if recommend_clicked:
            with st.spinner("Computing recommendations…"):
                try:
                    recs = engine.recommend(chosen_id, n=n_recs)
                    st.session_state.last_recs = recs
                    st.session_state.last_movie_id = chosen_id
                except ValueError as exc:
                    st.error(str(exc))
                    st.session_state.last_recs = None

        # ── Results ────────────────────────────────────────────────────────────
        show_recs = (
            st.session_state.last_recs is not None
            and st.session_state.last_movie_id == chosen_id
        )

        if show_recs:
            recs = st.session_state.last_recs
            if not recs:
                st.info("No recommendations found. Try a different movie or expand the genre filter.")
            else:
                st.markdown(
                    f"<p style='font-size:1.1rem;font-weight:700;color:#eeeef5;margin-bottom:.3rem'>"
                    f"Top {len(recs)} films similar to "
                    f"<span style='color:#e63946'>{chosen_title}</span></p>"
                    f"<p style='font-size:.78rem;color:#7e7e9a;margin-bottom:1.2rem'>"
                    f"Similarity scores measure textual feature overlap — not personal taste or model accuracy.</p>",
                    unsafe_allow_html=True,
                )

                col_a, col_b = st.columns(2, gap="medium")
                for i, rec in enumerate(recs):
                    target_col = col_a if i % 2 == 0 else col_b
                    with target_col:
                        with st.container(border=True):
                            _rec_card(
                                rank=i + 1,
                                title=rec.title,
                                year=rec.year,
                                genres=rec.genres,
                                overview=rec.overview,
                                score=rec.similarity_score,
                                explanation=rec.explanation,
                            )
        else:
            # Empty state
            st.markdown(
                "<div style='text-align:center;padding:4rem 1rem;'>"
                "<div style='font-size:3.5rem;margin-bottom:1rem'>🎬</div>"
                "<p style='font-size:1.05rem;color:#4a4a65;margin:0'>"
                "Select a movie above, then click "
                "<strong style='color:#e63946'>Find Similar Movies</strong>.</p>"
                "</div>",
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Browse
    # ══════════════════════════════════════════════════════════════════════════
    with tab_browse:
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

        b_left, b_right = st.columns([3, 1])
        with b_left:
            search_term = st.text_input(
                "Search", placeholder="Search by title, plot keyword, theme…",
                label_visibility="collapsed",
            )
        with b_right:
            st.markdown(
                f"<p style='text-align:right;color:#7e7e9a;font-size:.82rem;"
                f"padding-top:.6rem'>{len(filtered_df)} movies</p>",
                unsafe_allow_html=True,
            )

        if search_term:
            mask = (
                filtered_df["title"].str.contains(search_term, case=False, na=False)
                | filtered_df["overview"].str.contains(search_term, case=False, na=False)
                | filtered_df.get("keywords", pd.Series(dtype=str)).str.contains(
                    search_term, case=False, na=False
                )
            )
            browse_df = filtered_df[mask]
            st.caption(f"{len(browse_df)} result(s) for '{search_term}'")
        else:
            browse_df = filtered_df

        if browse_df.empty:
            st.info("No movies match your search. Try a broader term.")
        else:
            col1, col2 = st.columns(2, gap="medium")
            for i, (_, row) in enumerate(browse_df.iterrows()):
                target = col1 if i % 2 == 0 else col2
                with target:
                    with st.container(border=True):
                        _browse_card(
                            title=str(row["title"]),
                            year=row.get("year"),
                            genres=str(row.get("genres", "")),
                            overview=str(row.get("overview", "")),
                        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — How It Works
    # ══════════════════════════════════════════════════════════════════════════
    with tab_algo:
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

        st.markdown(
            "<p style='font-size:1.3rem;font-weight:800;color:#eeeef5;margin-bottom:.4rem'>"
            "How CineMatch Works</p>"
            "<p style='font-size:.88rem;color:#7e7e9a;margin-bottom:1.5rem'>"
            "CineMatch uses <strong style='color:#eeeef5'>content-based filtering</strong> "
            "— a classical AI technique that recommends movies based on their descriptive "
            "metadata. No user ratings or viewing history are required.</p>",
            unsafe_allow_html=True,
        )

        steps = [
            ("📂", "Load Movie Metadata",
             "Reads data/movies.csv — title, year, genres, overview, and keywords for 100 films."),
            ("🔗", "Combine Text Features",
             "Genres (repeated ×2 for extra weight), overview, and keywords are joined into "
             "a single feature string called a soup for each movie."),
            ("📐", "TF-IDF Vectorisation",
             "scikit-learn's TfidfVectorizer converts every soup into a sparse numerical vector. "
             "Words that are distinctive to one film receive higher weights than words that "
             "appear across the whole collection (e.g. 'the', 'and')."),
            ("📏", "Cosine Similarity",
             "For the selected movie, the cosine similarity between its TF-IDF vector and every "
             "other movie's vector is computed. Cosine similarity measures the angle between "
             "two vectors — 1.0 means identical; 0.0 means no shared features."),
            ("📊", "Rank & Filter",
             "All other movies are ranked by descending similarity. The selected movie itself is "
             "excluded (it would trivially score 1.0 against itself)."),
            ("🎬", "Return Results",
             "The top-N results are returned with similarity scores and plain-English explanations "
             "derived from shared genres and overlapping thematic keywords."),
        ]

        for icon, step_title, desc in steps:
            with st.container(border=True):
                sc1, sc2 = st.columns([1, 11])
                with sc1:
                    st.markdown(
                        f"<p style='font-size:1.6rem;margin:0;line-height:1'>{icon}</p>",
                        unsafe_allow_html=True,
                    )
                with sc2:
                    st.markdown(
                        f"<p style='font-size:.95rem;font-weight:700;color:#eeeef5;margin:0 0 3px'>"
                        f"{step_title}</p>"
                        f"<p style='font-size:.83rem;color:#7e7e9a;margin:0;line-height:1.55'>"
                        f"{desc}</p>",
                        unsafe_allow_html=True,
                    )

        st.divider()

        # Cosine formula
        st.markdown(
            "<p style='font-weight:700;color:#eeeef5;margin-bottom:.4rem'>Cosine Similarity Formula</p>",
            unsafe_allow_html=True,
        )
        st.latex(r"\text{similarity}(A,\, B) \;=\; \frac{A \cdot B}{\|A\| \times \|B\|}")
        st.caption(
            "A · B is the dot product of the two TF-IDF vectors. "
            "‖A‖ and ‖B‖ are their magnitudes. "
            "The result is always in [0, 1] for TF-IDF vectors with non-negative weights."
        )

        st.divider()

        comp_a, comp_b = st.columns(2, gap="large")
        with comp_a:
            with st.container(border=True):
                st.markdown(
                    "<p style='font-weight:700;color:#e63946;margin-bottom:.5rem'>"
                    "✅ Content-Based (this project)</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "- Uses movie metadata (genres, descriptions)\n"
                    "- No user data required\n"
                    "- Works for brand-new users\n"
                    "- Transparent and explainable\n"
                    "- Limited by metadata quality"
                )
        with comp_b:
            with st.container(border=True):
                st.markdown(
                    "<p style='font-weight:700;color:#7e7e9a;margin-bottom:.5rem'>"
                    "Collaborative Filtering</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "- Uses ratings from many users\n"
                    "- Finds hidden patterns in behaviour\n"
                    "- Cold-start problem for new items\n"
                    "- Requires substantial user data\n"
                    "- Less interpretable"
                )

        st.divider()
        st.info(
            "**Similarity ≠ Accuracy.** A cosine similarity score reflects how much the two "
            "movies share in their text features. It is not a measure of model training "
            "accuracy, nor a guarantee you will enjoy the recommendation."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — About
    # ══════════════════════════════════════════════════════════════════════════
    with tab_about:
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

        ab_l, ab_r = st.columns([3, 2], gap="large")

        with ab_l:
            st.markdown(
                "<p style='font-size:1.3rem;font-weight:800;color:#eeeef5;margin-bottom:1rem'>"
                "About CineMatch</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "**Project Title:** AI-Based Movie Recommendation System Using Content-Based Filtering\n\n"
                "**Subject:** Fundamentals of Artificial Intelligence (FAI)\n\n"
                "**Academic Level:** Second-Year B.Tech — Artificial Intelligence and Machine Learning"
            )
            st.divider()
            st.markdown("### Technologies")
            st.table(
                pd.DataFrame(
                    {
                        "Component": ["Language", "Recommendation", "Interface", "Data", "Testing"],
                        "Technology": [
                            "Python 3.13",
                            "scikit-learn TF-IDF + Cosine Similarity",
                            "Streamlit",
                            "Pandas",
                            "Pytest",
                        ],
                    }
                )
            )
            st.divider()
            st.markdown("### Dataset")
            st.markdown(
                "- **Type:** Curated sample dataset — 100 hand-selected films\n"
                "- **Genres:** 16 — Action, Sci-Fi, Romance, Comedy, Animation, Thriller, "
                "Drama, Fantasy, Horror, Musical, Crime, Biography, History, War, Music, Adventure\n"
                "- **Fields:** movie_id, title, year, genres, overview, keywords\n"
                "- **Offline:** Yes — no internet required at runtime\n"
                "- **Posters:** Not included — no external API required"
            )
            st.divider()
            st.markdown("### Limitations")
            st.markdown(
                "1. Small 100-movie dataset — less varied than production systems\n"
                "2. No personalisation — same movie always returns same results\n"
                "3. No collaborative signals, ratings, or user history\n"
                "4. Similarity depends on description quality\n"
                "5. Classical NLP + linear algebra only — no neural network\n"
                "6. Similarity scores are low (5–15%) due to diverse vocabulary across 100 films"
            )
            st.divider()
            st.markdown("### Future Scope")
            st.markdown(
                "- Expand to MovieLens or TMDB (tens of thousands of films)\n"
                "- Add collaborative filtering for a hybrid model\n"
                "- Integrate TMDB API for poster images (free key)\n"
                "- Deploy on Streamlit Community Cloud\n"
                "- Add user-preference tracking and evaluation metrics (Precision@K)"
            )

        with ab_r:
            with st.container(border=True):
                st.markdown(
                    "<p style='font-size:1.4rem;font-weight:900;color:#e63946;"
                    "letter-spacing:2px;margin-bottom:.3rem'>🎬 CineMatch</p>"
                    "<p style='font-size:.82rem;color:#7e7e9a;margin-bottom:1rem'>"
                    "Content-Based Movie Recommendation</p>",
                    unsafe_allow_html=True,
                )
                st.metric("Movies", 100)
                st.metric("Genres", 16)
                st.metric("Algorithm", "TF-IDF + Cosine")
                st.metric("Framework", "Streamlit")
                st.metric("Dataset", "Curated Local CSV")

            st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:.75rem;color:#4a4a65;line-height:1.7'>"
                "This project demonstrates classical AI content-based filtering. "
                "No neural network was trained. No user data is collected. "
                "No external APIs are required.</p>",
                unsafe_allow_html=True,
            )

    # ── Footer ──────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='margin-top:3rem;padding:1.2rem 0;border-top:1px solid #252535;"
        "text-align:center;font-size:.75rem;color:#4a4a65'>"
        "CineMatch · FAI Mini Project · B.Tech AI/ML 2nd Year · "
        "TF-IDF + Cosine Similarity · Streamlit + scikit-learn"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
