import html
import streamlit as st
from app import (
    DATABASE,
    check_movie_exists,
    explain_recommendation,
    get_available_movies,
    get_driver,
    get_gemini_client,
    get_recommendations,
)

# Page configuration
st.set_page_config(
    page_title="AI Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# Custom movie-themed styling: dark charcoal/navy backdrop, warm readable text, teal accents
CUSTOM_CSS = """
<style>
/* Main app background and base typography */
.stApp {
    background: linear-gradient(180deg, #0d131f 0%, #111827 100%);
    color: #e2e8f0;
}

/* Header typography */
h1 {
    color: #f8fafc !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem !important;
}

.subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    font-style: italic;
    margin-bottom: 2rem;
}

/* Form container styling */
[data-testid="stForm"] {
    background-color: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(45, 212, 191, 0.2);
    border-radius: 12px;
    padding: 1.5rem 1.75rem 1.25rem 1.75rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}

/* Make input label 'Enter a movie you like' white and readable */
[data-testid="stTextInput"] label,
[data-testid="stTextInput"] label p,
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] label p,
[data-testid="stWidgetLabel"] p {
    color: #ffffff !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}

/* Recommendation card styling */
.recommendation-card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
}

.movie-title {
    color: #2dd4bf;
    font-size: 1.35rem;
    font-weight: 700;
    margin-bottom: 1rem;
    border-bottom: 1px solid #334155;
    padding-bottom: 0.5rem;
}

.section-label {
    color: #38bdf8;
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
}

.evidence-box {
    background-color: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 1rem;
}

.evidence-item {
    font-size: 0.95rem;
    color: #cbd5e1;
    margin-bottom: 0.35rem;
}

.evidence-item:last-child {
    margin-bottom: 0;
}

.evidence-value {
    color: #f1f5f9;
    font-weight: 500;
}

.ai-box {
    background-color: rgba(45, 212, 191, 0.05);
    border-left: 3px solid #2dd4bf;
    border-radius: 4px;
    padding: 0.85rem 1rem;
}

.ai-text {
    font-size: 0.95rem;
    line-height: 1.6;
    color: #e2e8f0;
    margin: 0;
}

/* Available movies container */
.available-movies-box {
    background-color: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 10px;
    padding: 1.25rem;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
}

.available-movies-badge {
    display: inline-block;
    background-color: #0f172a;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 0.35rem 0.75rem;
    margin: 0.25rem 0.25rem;
    color: #cbd5e1;
    font-size: 0.9rem;
}

/* Custom button accent styling */
button[kind="primary"] {
    background-color: #0d9488 !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: background-color 0.2s ease;
}

button[kind="primary"]:hover {
    background-color: #14b8a6 !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Client caching to prevent reconnecting on every rerun
@st.cache_resource(show_spinner=False)
def init_driver():
    return get_driver()


@st.cache_resource(show_spinner=False)
def init_gemini():
    return get_gemini_client()


def render_recommendations(movie_title, recommendations, gemini_client):
    """Render the recommendation cards for a found movie."""
    st.subheader(f"Recommendations based on {movie_title}")

    for idx, rec in enumerate(recommendations, start=1):
        rec_title = rec.get("movie", "Unknown Title")
        shared_actors = rec.get("sharedActors", [])
        shared_genres = rec.get("sharedGenres", [])

        actors_str = ", ".join(shared_actors) if shared_actors else "None"
        genres_str = ", ".join(shared_genres) if shared_genres else "None"

        # Generate AI explanation safely
        try:
            explanation = explain_recommendation(
                movie_title,
                rec,
                client=gemini_client
            )
            ai_content = f"<p class=\"ai-text\">{html.escape(explanation)}</p>"
        except Exception:
            ai_content = (
                "<p class=\"ai-text\" style=\"color: #f59e0b;\">"
                "Explanation unavailable. The Neo4j recommendation is still valid."
                "</p>"
            )

        card_html = f"""
        <div class="recommendation-card">
            <div class="movie-title">{idx}. {html.escape(rec_title)}</div>
            <div class="section-label">Graph Evidence</div>
            <div class="evidence-box">
                <div class="evidence-item">
                    <strong>Shared actors:</strong> <span class="evidence-value">{html.escape(actors_str)}</span>
                </div>
                <div class="evidence-item">
                    <strong>Shared genres:</strong> <span class="evidence-value">{html.escape(genres_str)}</span>
                </div>
            </div>
            <div class="section-label">AI Explanation</div>
            <div class="ai-box">
                {ai_content}
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)


def process_movie_search(movie_title):
    """Validate movie against Neo4j and return status, recommendations, or available titles."""
    try:
        driver = init_driver()
        gemini_client = init_gemini()
    except Exception:
        return {"status": "init_error", "message": "Service initialization failed. Please check your database and API configuration."}

    try:
        with driver.session(database=DATABASE) as session:
            # Check whether the movie exists
            if not check_movie_exists(session, movie_title):
                # Retrieve actual available movie titles from Neo4j
                available_movies = get_available_movies(session)
                return {
                    "status": "not_found",
                    "movie_title": movie_title,
                    "available_movies": available_movies
                }

            # Retrieve recommendations
            recommendations = get_recommendations(session, movie_title)
            return {
                "status": "success",
                "movie_title": movie_title,
                "recommendations": recommendations,
                "gemini_client": gemini_client
            }
    except Exception:
        return {"status": "db_error", "message": "Unable to communicate with the Neo4j database. Please verify your connection and try again."}


def main():
    st.markdown("<h1>AI Movie Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown(
        "<div class=\"subtitle\">Discover movies through connections in a knowledge graph.</div>",
        unsafe_allow_html=True
    )

    # Initialize session state for searched movie or selected available movie
    if "search_target" not in st.session_state:
        st.session_state["search_target"] = ""
    if "available_options" not in st.session_state:
        st.session_state["available_options"] = []

    # Form prevents automatic queries on input change or normal Streamlit reruns
    with st.form(key="recommendation_form"):
        movie_input = st.text_input(
            label="Enter a movie you like",
            placeholder="e.g. 3 Idiots, Dangal"
        )
        submit_button = st.form_submit_button(
            label="Get Recommendations",
            type="primary"
        )

    # Determine movie to run recommendation for
    target_to_search = None
    if submit_button:
        cleaned_input = movie_input.strip()
        if not cleaned_input:
            st.warning("Please enter a movie title.")
            st.session_state["search_target"] = ""
            st.session_state["available_options"] = []
            return
        target_to_search = cleaned_input
        st.session_state["search_target"] = cleaned_input
    elif st.session_state.get("search_target"):
        target_to_search = st.session_state["search_target"]

    if target_to_search:
        with st.spinner("Finding recommendations and analyzing connections..."):
            result = process_movie_search(target_to_search)

        if result["status"] == "init_error" or result["status"] == "db_error":
            st.error(result["message"])
            return

        if result["status"] == "not_found":
            st.info(
                f"'{result['movie_title']}' is not in our movie database. "
                "Please choose a movie from our available collection below."
            )

            available_movies = result.get("available_movies", [])
            st.session_state["available_options"] = available_movies

            if available_movies:
                badges_html = "".join(
                    f'<span class="available-movies-badge">{html.escape(m)}</span>'
                    for m in available_movies
                )
                st.markdown(
                    f"""
                    <div class="available-movies-box">
                        <div class="section-label" style="color: #2dd4bf; margin-bottom: 0.6rem;">
                            Available Movies in Database ({len(available_movies)})
                        </div>
                        <div>{badges_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("##### Or pick a title directly to search:")
                with st.form(key="available_selection_form"):
                    selected_movie = st.selectbox(
                        label="Select from available movies",
                        options=available_movies
                    )
                    select_submit = st.form_submit_button(
                        label="Get Recommendations for Selected Movie",
                        type="primary"
                    )

                if select_submit and selected_movie:
                    st.session_state["search_target"] = selected_movie
                    st.rerun()
            else:
                st.info("No movies were found in the database.")
            return

        # Case: Successful movie found
        st.session_state["available_options"] = []
        recommendations = result.get("recommendations", [])
        if not recommendations:
            st.info(
                f"No recommendations found for '{result['movie_title']}'. "
                "Our database may not contain enough actor or genre connections for this movie."
            )
            return

        render_recommendations(
            result["movie_title"],
            recommendations,
            result["gemini_client"]
        )


if __name__ == "__main__":
    main()
