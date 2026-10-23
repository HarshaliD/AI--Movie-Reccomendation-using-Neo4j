"""Intent Router for Movie + Employee Knowledge Graph Assistant.

Hybrid rules-first router that classifies user input into:
- 'movie'
- 'employee'
- 'general'
- 'ambiguous'

Rules-first architecture:
- Zero LLM calls for classification.
- Greetings and help do not trigger database queries.
- Unrecognized questions NEVER fall through to the movie handler.
- Clear disambiguation when signals are mixed or underspecified.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set


@dataclass
class RouteResult:
    intent: str  # 'movie', 'employee', 'general', 'ambiguous'
    confidence: float
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


# Known movie titles in database for exact/phrase matching
KNOWN_MOVIES = {
    "3 idiots",
    "chhichhore",
    "dangal",
    "kahaani",
    "lage raho munna bhai",
    "munna bhai m.b.b.s.",
    "munna bhai mbbs",
    "pk",
    "rab ne bana di jodi",
    "sanju",
    "stree",
}

# Greetings, gratitude, and farewell phrases
GREETING_TOKENS = {
    "hi",
    "hello",
    "hey",
    "hola",
    "namaste",
    "good morning",
    "good afternoon",
    "good evening",
    "greetings",
    "howdy",
}

COURTESY_TOKENS = {
    "thanks",
    "thank you",
    "thx",
    "thank you so much",
    "appreciate it",
    "bye",
    "goodbye",
    "see you",
    "have a nice day",
    "take care",
}

HELP_TOKENS = {
    "help",
    "help me",
    "what can you do",
    "how to use",
    "commands",
    "options",
    "guide",
    "capabilities",
}

# Core domain keyword indicators
MOVIE_KEYWORDS = {
    "movie",
    "movies",
    "film",
    "films",
    "cinema",
    "recommend",
    "recommendation",
    "recommendations",
    "actor",
    "actors",
    "actress",
    "actresses",
    "genre",
    "genres",
    "starring",
    "featuring",
    "director",
    "directed by",
    "box office",
    "watch next",
    "bollywood",
}

EMPLOYEE_KEYWORDS = {
    "employee",
    "employees",
    "emp",
    "staff",
    "worker",
    "workers",
    "colleague",
    "colleagues",
    "manager",
    "managers",
    "reports to",
    "reporting to",
    "report to",
    "direct reports",
    "department",
    "departments",
    "designation",
    "designations",
    "salary",
    "salaries",
    "earning",
    "earns",
    "earn",
    "lakh",
    "lakhs",
    "lpa",
    "inr",
    "joining date",
    "joined",
    "hire",
    "hired",
    "skills",
    "skill",
}

KNOWN_DEPARTMENTS = {
    "engineering",
    "data science",
    "human resources",
    "hr",
    "finance",
    "marketing",
    "product",
}

KNOWN_LOCATIONS = {
    "bengaluru",
    "bangalore",
    "hyderabad",
    "pune",
    "mumbai",
    "gurugram",
    "gurgaon",
}

COMMON_TECH_SKILLS = {
    "python",
    "sql",
    "react",
    "docker",
    "kubernetes",
    "aws",
    "fastapi",
    "django",
    "tableau",
    "pytorch",
    "mlops",
    "pandas",
    "scikit-learn",
    "typescript",
    "javascript",
    "terraform",
}

# Regex for employee IDs: EMP followed by digits
EMP_ID_PATTERN = re.compile(r"\bemp\d{3}\b", re.IGNORECASE)

# Underspecified ambiguous query patterns
AMBIGUOUS_PATTERNS = [
    r"^show me the best ones?$",
    r"^find (them|it|all)$",
    r"^show (them|it|all)$",
    r"^list all$",
    r"^what do you have\??$",
    r"^tell me more$",
    r"^give me options?$",
]


def normalize_text(text: str) -> str:
    """Normalize input text by lowering, stripping, and reducing excessive whitespace."""
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    return cleaned


def has_token_match(normalized: str, token_set: Set[str]) -> bool:
    """Check if any token or multi-word phrase from token_set exists as a word or phrase."""
    for token in token_set:
        pattern = r"\b" + re.escape(token) + r"\b"
        if re.search(pattern, normalized):
            return True
    return False


def count_token_matches(normalized: str, token_set: Set[str]) -> int:
    """Count occurrences of matching tokens with word boundaries."""
    count = 0
    for token in token_set:
        pattern = r"\b" + re.escape(token) + r"\b"
        if re.search(pattern, normalized):
            count += 1
    return count


def route_intent(user_input: str, custom_known_movies: Optional[Set[str]] = None) -> RouteResult:
    """Route a user question to 'movie', 'employee', 'general', or 'ambiguous'.

    Args:
        user_input: Raw query string from the user.
        custom_known_movies: Optional set of known movie titles to check against.

    Returns:
        RouteResult containing intent classification and rationale.
    """
    if not user_input or not user_input.strip():
        return RouteResult(
            intent="general",
            confidence=1.0,
            reason="Empty input provided.",
            details={"type": "empty"},
        )

    norm = normalize_text(user_input)
    punct_stripped = re.sub(r"^[^\w\s]+|[^\w\s]+$", "", norm).strip()
    movies_set = set(custom_known_movies) if custom_known_movies else KNOWN_MOVIES

    # 1. Check explicit Ambiguous patterns
    for pat in AMBIGUOUS_PATTERNS:
        if re.match(pat, punct_stripped):
            return RouteResult(
                intent="ambiguous",
                confidence=0.9,
                reason="Query is underspecified without domain context.",
                details={"pattern": pat},
            )

    # 2. Check Greetings / Gratitude / Help (without domain keywords)
    is_greeting = punct_stripped in GREETING_TOKENS or any(
        re.match(rf"^{re.escape(g)}\b", punct_stripped) for g in GREETING_TOKENS
    )
    is_courtesy = punct_stripped in COURTESY_TOKENS
    is_help = punct_stripped in HELP_TOKENS or "what can you do" in punct_stripped or "how does this work" in punct_stripped

    has_emp_id = bool(EMP_ID_PATTERN.search(norm))
    emp_kw_count = count_token_matches(norm, EMPLOYEE_KEYWORDS)
    movie_kw_count = count_token_matches(norm, MOVIE_KEYWORDS)

    if (is_greeting or is_courtesy or is_help) and not has_emp_id and emp_kw_count == 0 and movie_kw_count == 0:
        return RouteResult(
            intent="general",
            confidence=1.0,
            reason="User message is a greeting, courtesy, or help request.",
            details={
                "is_greeting": is_greeting,
                "is_courtesy": is_courtesy,
                "is_help": is_help,
            },
        )

    # 3. Employee Signals Scoring
    emp_score = 0
    emp_signals = []

    if has_emp_id:
        emp_score += 4
        emp_signals.append("employee_id_pattern")

    if emp_kw_count > 0:
        emp_score += emp_kw_count * 2
        emp_signals.append(f"employee_keywords_{emp_kw_count}")

    dept_matches = [d for d in KNOWN_DEPARTMENTS if re.search(rf"\b{re.escape(d)}\b", norm)]
    if dept_matches:
        if re.search(r"\b(who works in|how many|headcount|team in)\b", norm) or emp_kw_count > 0:
            emp_score += 3
            emp_signals.append(f"department_query_{dept_matches[0]}")
        else:
            emp_score += 1
            emp_signals.append(f"department_name_{dept_matches[0]}")

    if re.search(r"\b(who is .*('s)? manager|who reports to|direct reports|earning above|earning below|salary above|salary below)\b", norm):
        emp_score += 4
        emp_signals.append("employee_specific_phrase")

    if re.search(r"\b(which employees know|employees with|who knows|skills? in)\b", norm):
        emp_score += 3
        emp_signals.append("employee_skills_phrase")

    skill_matches = [s for s in COMMON_TECH_SKILLS if re.search(rf"\b{re.escape(s)}\b", norm)]
    if skill_matches:
        if emp_kw_count > 0 or has_emp_id or re.search(r"\b(know|knows|skill|skills|expert|developer|engineer)\b", norm):
            emp_score += 2 + len(skill_matches)
            emp_signals.append(f"tech_skills_{len(skill_matches)}")

    loc_matches = [loc for loc in KNOWN_LOCATIONS if re.search(rf"\b{re.escape(loc)}\b", norm)]
    if loc_matches and (emp_kw_count > 0 or has_emp_id or re.search(r"\b(employees?|staff|workers?)\b", norm)):
        emp_score += 2
        emp_signals.append(f"location_emp_{loc_matches[0]}")

    # 4. Movie Signals Scoring
    movie_score = 0
    movie_signals = []

    if movie_kw_count > 0:
        movie_score += movie_kw_count * 2
        movie_signals.append(f"movie_keywords_{movie_kw_count}")

    for title in movies_set:
        if re.search(rf"\b{re.escape(title)}\b", norm):
            if title == "sanju" and (has_emp_id or emp_kw_count > 0):
                continue
            movie_score += 4
            movie_signals.append(f"known_movie_{title}")
            break

    if re.search(r"\b(recommend( me)? movies?|movies? like|movies? featuring|movies? starring|similar to)\b", norm):
        movie_score += 3
        movie_signals.append("movie_recommender_phrase")

    if punct_stripped in movies_set:
        movie_score += 5
        movie_signals.append("exact_movie_title")

    # 5. Evaluate Scores & Conflict Resolution
    if emp_score > 0 and movie_score > 0:
        if (emp_kw_count > 0 or has_emp_id) and movie_kw_count > 0:
            return RouteResult(
                intent="ambiguous",
                confidence=0.8,
                reason="Query contains signals from both movie and employee domains.",
                details={"emp_score": emp_score, "movie_score": movie_score, "emp_signals": emp_signals, "movie_signals": movie_signals},
            )

        if emp_score >= movie_score * 3.0:
            return RouteResult(
                intent="employee",
                confidence=min(0.85, 0.5 + 0.1 * emp_score),
                reason="Predominant employee intent despite movie term.",
                details={"emp_score": emp_score, "movie_score": movie_score, "signals": emp_signals},
            )
        elif movie_score >= emp_score * 3.0:
            return RouteResult(
                intent="movie",
                confidence=min(0.85, 0.5 + 0.1 * movie_score),
                reason="Predominant movie intent despite employee term.",
                details={"movie_score": movie_score, "emp_score": emp_score, "signals": movie_signals},
            )
        else:
            return RouteResult(
                intent="ambiguous",
                confidence=0.7,
                reason="Query contains conflicting signals from both movie and employee domains.",
                details={"emp_score": emp_score, "movie_score": movie_score, "emp_signals": emp_signals, "movie_signals": movie_signals},
            )

    if emp_score >= 2:
        return RouteResult(
            intent="employee",
            confidence=min(0.95, 0.6 + 0.08 * emp_score),
            reason=f"Employee query detected based on: {', '.join(emp_signals)}",
            details={"emp_score": emp_score, "signals": emp_signals},
        )

    if movie_score >= 2:
        return RouteResult(
            intent="movie",
            confidence=min(0.95, 0.6 + 0.08 * movie_score),
            reason=f"Movie query detected based on: {', '.join(movie_signals)}",
            details={"movie_score": movie_score, "signals": movie_signals},
        )

    if punct_stripped in movies_set:
        return RouteResult(
            intent="movie",
            confidence=0.9,
            reason=f"Exact match for known movie title: '{punct_stripped}'.",
            details={"movie_title": punct_stripped},
        )

    return RouteResult(
        intent="ambiguous",
        confidence=0.5,
        reason="Unrecognized query. Intent cannot be determined reliably from available rules.",
        details={"norm_query": norm},
    )
