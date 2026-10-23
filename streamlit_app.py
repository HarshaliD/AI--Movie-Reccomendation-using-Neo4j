"""AI Movie & Employee Knowledge Graph Assistant.

Unified Streamlit application supporting:
1. Movie Recommendations via Neo4j graph relationships + Gemini AI explanations.
2. Employee Knowledge Graph queries (lookups, reporting hierarchy, department/location/skill/salary filtering).
3. Zero-LLM rules-first intent routing for instant greetings, help, and clarification.
"""

import html
import os
from pathlib import Path
import re
import sys
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
MOVIE_REC_DIR = ROOT_DIR / "movie-recommendation"
if str(MOVIE_REC_DIR) not in sys.path:
    sys.path.insert(0, str(MOVIE_REC_DIR))

from app import (
    DATABASE,
    check_movie_exists,
    explain_recommendation,
    get_available_movies,
    get_driver,
    get_gemini_client,
    get_recommendations,
)
from employee_queries import _format_inr, handle_employee_query
from intent_router import KNOWN_MOVIES, route_intent

# Page configuration
st.set_page_config(
    page_title="AI Movie & Employee Knowledge Graph Assistant",
    page_icon="🎬",
    layout="centered",
)

CUSTOM_CSS = """
<style>
.stApp {
    background: linear-gradient(180deg, #0d131f 0%, #111827 100%);
    color: #e2e8f0;
}

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
    margin-bottom: 1.5rem;
}

.domain-badge-container {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}

.domain-badge {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 9999px;
    padding: 0.35rem 0.85rem;
    font-size: 0.82rem;
    font-weight: 500;
    color: #cbd5e1;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}

.domain-badge.active {
    border-color: #2dd4bf;
    color: #2dd4bf;
    background-color: rgba(45, 212, 191, 0.08);
}

[data-testid="stForm"] {
    background-color: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(45, 212, 191, 0.25);
    border-radius: 12px;
    padding: 1.5rem 1.75rem 1.25rem 1.75rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}

[data-testid="stTextInput"] label,
[data-testid="stTextInput"] label p,
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] label p,
[data-testid="stWidgetLabel"] p {
    color: #ffffff !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}

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

.emp-card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.35rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
}

.emp-name {
    color: #38bdf8;
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.emp-role {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-bottom: 0.85rem;
}

.emp-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.6rem;
    background-color: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.85rem;
}

.emp-grid-item {
    font-size: 0.9rem;
    color: #cbd5e1;
}

.emp-grid-label {
    color: #64748b;
    font-size: 0.8rem;
    text-transform: uppercase;
    font-weight: 600;
}

.emp-badge {
    display: inline-block;
    background-color: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 6px;
    padding: 0.2rem 0.55rem;
    margin: 0.15rem 0.2rem 0.15rem 0;
    color: #7dd3fc;
    font-size: 0.82rem;
}

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

.assistant-card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.assistant-title {
    color: #2dd4bf;
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
}

.example-query {
    background-color: #0f172a;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 0.4rem 0.8rem;
    margin: 0.25rem 0;
    color: #cbd5e1;
    font-family: monospace;
    font-size: 0.88rem;
    display: block;
}

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


@st.cache_resource(show_spinner=False)
def init_driver():
    return get_driver()


@st.cache_resource(show_spinner=False)
def init_gemini():
    return get_gemini_client()


def render_movie_recommendations(movie_title, recommendations, gemini_client):
    """Render the recommendation cards for a found movie with shared evidence and AI explanation."""
    st.subheader(f"Recommendations based on {movie_title}")

    for idx, rec in enumerate(recommendations, start=1):
        rec_title = rec.get("movie", "Unknown Title")
        shared_actors = rec.get("sharedActors", [])
        shared_genres = rec.get("sharedGenres", [])

        actors_str = ", ".join(shared_actors) if shared_actors else "None"
        genres_str = ", ".join(shared_genres) if shared_genres else "None"

        try:
            explanation = explain_recommendation(
                movie_title,
                rec,
                client=gemini_client,
            )
            ai_content = f'<p class="ai-text">{html.escape(explanation)}</p>'
        except Exception:
            ai_content = (
                '<p class="ai-text" style="color: #f59e0b;">'
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


def render_employee_profile(emp):
    """Render an individual employee's detailed profile card."""
    skills_html = "".join(f'<span class="emp-badge">{html.escape(s)}</span>' for s in emp.get("skills", []))
    card_html = f"""
    <div class="emp-card">
        <div class="emp-name">{html.escape(emp['name'])} <span style="font-size: 0.9rem; color: #64748b;">({html.escape(emp['employee_id'])})</span></div>
        <div class="emp-role">{html.escape(emp.get('designation', ''))} &bull; <strong style="color: #2dd4bf;">{html.escape(emp.get('department', ''))}</strong></div>
        <div class="emp-grid">
            <div class="emp-grid-item"><div class="emp-grid-label">Location</div>{html.escape(emp.get('location', 'N/A'))}</div>
            <div class="emp-grid-item"><div class="emp-grid-label">Salary</div>{html.escape(_format_inr(emp.get('salary')))}</div>
            <div class="emp-grid-item"><div class="emp-grid-label">Age</div>{emp.get('age', 'N/A')} yrs</div>
            <div class="emp-grid-item"><div class="emp-grid-label">Joining Date</div>{html.escape(str(emp.get('joining_date', 'N/A')))}</div>
            <div class="emp-grid-item" style="grid-column: 1 / -1;"><div class="emp-grid-label">Manager</div>{html.escape(emp.get('manager_name') or 'Top-level Executive (None)')}</div>
        </div>
        <div class="section-label" style="font-size: 0.78rem;">Skills</div>
        <div>{skills_html or '<span style="color: #64748b;">None listed</span>'}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_employee_results(result):
    """Render structured employee query results based on query_type."""
    query_type = result.get("query_type")
    data = result.get("data")
    msg = result.get("message", "")

    if result.get("status") == "empty" or not data:
        st.info(msg)
        return

    if query_type == "employee_profile":
        lookup = data
        if lookup.get("status") == "exact" and lookup.get("employee"):
            render_employee_profile(lookup["employee"])
        elif lookup.get("status") == "multiple":
            st.warning(msg)
            candidates = lookup.get("candidates", [])
            for c in candidates:
                render_employee_profile(c)

    elif query_type == "manager_lookup":
        st.success(msg)
        emp = data.get("employee")
        mgr = data.get("manager")
        if emp:
            st.markdown("##### Employee Details")
            render_employee_profile(emp)
        if mgr:
            st.markdown("##### Manager Profile")
            render_employee_profile(mgr)

    elif query_type == "direct_reports":
        st.success(msg)
        reports = data.get("reports", [])
        if reports:
            table_rows = []
            for r in reports:
                table_rows.append({
                    "Employee ID": r["employee_id"],
                    "Name": r["name"],
                    "Designation": r["designation"],
                    "Department": r.get("department", "N/A"),
                    "Location": r["location"],
                    "Salary": _format_inr(r.get("salary")),
                    "Skills": ", ".join(r.get("skills", [])),
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    elif query_type in ("headcount", "headcount_all"):
        st.success(msg)
        df_counts = pd.DataFrame(data)
        df_counts.rename(columns={"department": "Department", "count": "Employee Count"}, inplace=True)
        st.dataframe(df_counts, use_container_width=True)

    elif query_type == "filtered_list":
        st.success(msg)
        records = data
        table_rows = []
        for r in records:
            table_rows.append({
                "ID": r["employee_id"],
                "Name": r["name"],
                "Role": r["designation"],
                "Department": r.get("department", "N/A"),
                "Location": r["location"],
                "Salary": _format_inr(r.get("salary")),
                "Manager": r.get("manager_name") or "None",
                "Skills": ", ".join(r.get("skills", [])),
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    else:
        st.write(msg)


def extract_movie_title_from_query(query: str, known_movies: set) -> str:
    """Extract candidate movie title if query contains phrases like 'Recommend movies like X'."""
    clean = query.strip()
    norm = clean.lower()

    for title in known_movies:
        if title in norm:
            return title.title()

    patterns = [
        r"(?:recommend\s+movies?\s+like|similar\s+to|movies?\s+featuring|movies?\s+starring)\s+([a-zA-Z0-9\s\.\:\'\-]+)",
        r"(?:recommend|show)\s+([a-zA-Z0-9\s\.\:\'\-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, clean, re.IGNORECASE)
        if m:
            extracted = m.group(1).strip()
            if extracted.lower() not in ("movies", "movie", "films", "film", "them"):
                return extracted

    return clean


def main():
    st.markdown("<h1>AI Knowledge Graph Assistant</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Intelligent Movie Recommendations and Employee Insights powered by Neo4j and Gemini</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="domain-badge-container">
            <span class="domain-badge active">🎬 Movie Recommender (Neo4j + Gemini)</span>
            <span class="domain-badge active">👥 Employee Knowledge Graph (Neo4j)</span>
            <span class="domain-badge active">⚡ Rules-First Intent Router</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "user_query" not in st.session_state:
        st.session_state["user_query"] = ""
    if "submitted_query" not in st.session_state:
        st.session_state["submitted_query"] = ""
    if "available_options" not in st.session_state:
        st.session_state["available_options"] = []

    with st.form(key="unified_query_form"):
        user_input = st.text_input(
            label="Ask a question about movies, employees, or skills:",
            placeholder="e.g. Recommend movies like 3 Idiots, Who works in Data Science?, Who is EMP012's manager?",
        )
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_button = st.form_submit_button(label="Search / Ask", type="primary")

    active_query = None
    if submit_button:
        cleaned = user_input.strip()
        if not cleaned:
            st.warning("Please enter a question or topic.")
            st.session_state["submitted_query"] = ""
            return
        active_query = cleaned
        st.session_state["submitted_query"] = cleaned
    elif st.session_state.get("submitted_query"):
        active_query = st.session_state["submitted_query"]

    if not active_query:
        st.markdown(
            """
            <div class="assistant-card">
                <div class="assistant-title">Try asking questions like:</div>
                <div class="section-label">🎬 Movie Recommendations</div>
                <span class="example-query">Recommend movies like 3 Idiots</span>
                <span class="example-query">Dangal</span>
                <div class="section-label" style="margin-top: 0.75rem;">👥 Employee Information</div>
                <span class="example-query">Who works in Data Science?</span>
                <span class="example-query">Who is EMP012's manager?</span>
                <span class="example-query">Which employees know Python and SQL?</span>
                <span class="example-query">List Hyderabad employees earning above 12 lakh</span>
                <span class="example-query">How many employees work in Engineering?</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    route = route_intent(active_query)

    # 1. GENERAL INTENT
    if route.intent == "general":
        st.markdown(
            f"""
            <div class="assistant-card">
                <div class="assistant-title">Hello! I am your Knowledge Graph Assistant 👋</div>
                <p style="color: #e2e8f0; line-height: 1.6;">
                    I can help you explore two connected domains stored in Neo4j AuraDB:
                </p>
                <div class="section-label">🎬 Movie Domain</div>
                <p style="color: #cbd5e1; font-size: 0.95rem; margin-bottom: 0.5rem;">
                    Find movie recommendations based on shared actors and genres, accompanied by Gemini AI explanations.
                </p>
                <span class="example-query">Recommend movies like 3 Idiots</span>
                <span class="example-query">Show movies like Dangal</span>
                <div class="section-label" style="margin-top: 1rem;">👥 Employee Domain</div>
                <p style="color: #cbd5e1; font-size: 0.95rem; margin-bottom: 0.5rem;">
                    Query organizational structure, departments, managers, skill sets, locations, and salaries.
                </p>
                <span class="example-query">Who works in Data Science?</span>
                <span class="example-query">Who is EMP012's manager?</span>
                <span class="example-query">Which employees know Python and SQL?</span>
                <span class="example-query">List Hyderabad employees earning above 12 lakh</span>
                <span class="example-query">How many employees work in Engineering?</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # 2. AMBIGUOUS INTENT
    if route.intent == "ambiguous":
        st.markdown(
            f"""
            <div class="assistant-card" style="border-color: #f59e0b;">
                <div class="assistant-title" style="color: #f59e0b;">Could you please clarify your request?</div>
                <p style="color: #e2e8f0; line-height: 1.6;">
                    {html.escape(route.reason)}
                </p>
                <p style="color: #cbd5e1; font-size: 0.92rem;">
                    Please specify whether you are inquiring about a <strong>movie recommendation</strong> or an <strong>employee query</strong>. For example:
                </p>
                <span class="example-query">Recommend movies like 3 Idiots</span>
                <span class="example-query">Who works in Engineering?</span>
                <span class="example-query">Who is EMP001's manager?</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # 3. MOVIE INTENT
    if route.intent == "movie":
        candidate_title = extract_movie_title_from_query(active_query, KNOWN_MOVIES)

        try:
            driver = init_driver()
            gemini_client = init_gemini()
        except Exception:
            st.error("Service initialization failed. Please check your database and API configuration.")
            return

        with st.spinner(f"Analyzing connections for '{candidate_title}' in Neo4j..."):
            try:
                with driver.session(database=DATABASE) as session:
                    if not check_movie_exists(session, candidate_title):
                        available_movies = get_available_movies(session)
                        st.info(
                            f"'{candidate_title}' is not in our movie database. "
                            "Please choose a movie from our available collection below."
                        )
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
                                unsafe_allow_html=True,
                            )

                            st.markdown("##### Or pick a title directly to search:")
                            with st.form(key="available_selection_form"):
                                selected_movie = st.selectbox(
                                    label="Select from available movies",
                                    options=available_movies,
                                )
                                select_submit = st.form_submit_button(
                                    label="Get Recommendations for Selected Movie",
                                    type="primary",
                                )

                            if select_submit and selected_movie:
                                st.session_state["submitted_query"] = selected_movie
                                st.rerun()
                        else:
                            st.info("No movies were found in the database.")
                        return

                    recommendations = get_recommendations(session, candidate_title)

            except Exception:
                st.error("Unable to communicate with the Neo4j database. Please verify your connection and try again.")
                return

        if not recommendations:
            st.info(
                f"No recommendations found for '{candidate_title}'. "
                "Our database may not contain enough actor or genre connections for this movie."
            )
            return

        render_movie_recommendations(candidate_title, recommendations, gemini_client)
        return

    # 4. EMPLOYEE INTENT
    if route.intent == "employee":
        try:
            driver = init_driver()
        except Exception:
            st.error("Database connection failed. Please check your Neo4j configuration.")
            return

        with st.spinner("Querying employee knowledge graph..."):
            try:
                with driver.session(database=DATABASE) as session:
                    result = handle_employee_query(active_query, session)
            except Exception:
                st.error("Failed to execute employee graph query. Please check database connectivity.")
                return

        render_employee_results(result)
        return


if __name__ == "__main__":
    main()
