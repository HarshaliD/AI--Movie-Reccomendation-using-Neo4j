# Simple AI Movie Recommendation System

## 1. Project Overview

This project implements a **simple AI-based Movie Recommendation System** using Neo4j, Python, Gemini LLM, and Streamlit.

The system recommends up to three movies based on shared actors or genres with a movie selected by the user. Neo4j is used to store the movie data as a graph and retrieve recommendations, while Gemini generates simple, evidence-based explanations for the recommendations. Streamlit provides the user interface.

**The implementation strictly follows the scope of the assignment.** It uses basic graph modelling, Cypher queries, relationship-based recommendation logic, and LLM-generated explanations. No additional techniques outside the assignment scope have been introduced.

## 2. Technology Stack

- **Neo4j AuraDB:** Stores movie data as a graph and retrieves recommendations.
- **Cypher:** Queries the graph and identifies movies with shared actors or genres.
- **Python:** Implements the application logic and connects the components.
- **Google Gemini API:** Generates explanations based on the retrieved graph evidence.
- **Streamlit:** Provides a simple interactive user interface.
- **python-dotenv:** Loads configuration and credentials from environment variables.

## 3. Graph Data Model

The application represents the movie dataset using three node types:

| Node | Properties |
|---|---|
| `Movie` | `title`, `year` |
| `Actor` | `name` |
| `Genre` | `name` |

The graph uses two relationship types:

- `(Actor)-[:ACTED_IN]->(Movie)` connects actors to movies.
- `(Movie)-[:IN_GENRE]->(Genre)` connects movies to genres.

This structure allows the application to retrieve movies through their relationships rather than relying on text similarity.

The project uses a small, manually defined dataset containing movies, actors, and genres. The dataset is kept intentionally limited to demonstrate the required graph-based recommendation logic.

## 4. Recommendation Logic

When a user enters a movie title, the application follows these steps:

1. Checks whether the entered movie exists in Neo4j.
2. Retrieves other movies connected to the selected movie through shared actors or genres.
3. Collects the shared actor names and genre names as supporting evidence.
4. Ranks the candidate movies using the number of shared actors and genres.
5. Returns a maximum of three recommendations.
6. Passes the retrieved evidence to Gemini to generate an explanation for each recommendation.

The recommendation logic is based on graph relationships. It does not calculate plot similarity or predict a user's personal preferences.

## 5. Key Cypher Query

The following query performs the main recommendation retrieval operation:

```cypher
MATCH (selected:Movie {title: $movie_title})
MATCH (recommended:Movie)
WHERE recommended <> selected

OPTIONAL MATCH
    (selected)<-[:ACTED_IN]-(actor:Actor)-[:ACTED_IN]->(recommended)

OPTIONAL MATCH
    (selected)-[:IN_GENRE]->(genre:Genre)<-[:IN_GENRE]-(recommended)

WITH recommended,
     collect(DISTINCT actor.name) AS sharedActors,
     collect(DISTINCT genre.name) AS sharedGenres

WHERE size(sharedActors) > 0 OR size(sharedGenres) > 0

RETURN recommended.title AS movie,
       sharedActors,
       sharedGenres

ORDER BY size(sharedActors) + size(sharedGenres) DESC
LIMIT 3
```

### How the query works

- **Movie selection:** Identifies the movie entered by the user using the `$movie_title` parameter.
- **Candidate selection:** Finds other movies and excludes the selected movie from the results.
- **Actor matching:** Finds actors who acted in both the selected movie and a candidate movie.
- **Genre matching:** Finds genres associated with both movies.
- **Evidence aggregation:** Uses `collect(DISTINCT ...)` to collect unique shared actor and genre names.
- **Filtering:** Keeps candidates that share at least one actor or genre with the selected movie.
- **Ranking:** Sorts candidates by the combined count of shared actors and genres.
- **Limiting:** Returns at most three recommendations.

`OPTIONAL MATCH` allows a candidate to qualify through either actors or genres, even if it has no shared connection of the other type.

The query returns not only the recommended movie titles but also the evidence supporting each recommendation. This evidence is subsequently used by the Python application and Gemini.

**Note:** The ranking is a simple count of shared actors and genres. It is not a machine-learning score or a guarantee that a user will enjoy a recommended movie.

## 6. Role of the Gemini LLM

Neo4j is responsible for finding the recommendations. Gemini is used only to explain the results.

For each recommended movie, the application provides Gemini with:

- The title of the selected movie.
- The title of the recommended movie.
- The shared actors retrieved from Neo4j.
- The shared genres retrieved from Neo4j.

Gemini uses this information to generate a short, readable explanation of why the movie was recommended.

The prompt instructs the model to use the supplied evidence and avoid inventing movie connections or unsupported facts.

This separation keeps the recommendation retrieval deterministic and graph-based, while the LLM makes the results easier for users to understand.

## 7. Application Architecture

```text
User
 |
 v
Streamlit UI
 |
 v
Python Application Logic
 |
 +--------------------------+
 |                          |
 v                          v
Neo4j AuraDB             Gemini API
 |                          ^
 v                          |
Recommended Movies          |
and Shared Evidence --------+
 |
 v
Recommendations with Explanations
 |
 v
Displayed in Streamlit
```

### Main files

**`app.py`**

Contains the core application logic, including:

- Neo4j driver configuration and database access.
- Movie existence validation.
- Retrieval of available movie titles.
- The main Cypher recommendation query.
- Processing of recommendation results.
- Gemini-based explanation generation.
- A command-line execution path for testing the core functionality.

**`streamlit_app.py`**

Contains the user interface, including:

- Movie title input and selection.
- Recommendation retrieval.
- Display of recommended movies and their supporting evidence.
- Display of Gemini-generated explanations.
- Handling of invalid movie titles and empty results.
- User-facing error messages for common failures.

## 8. User Interface and Error Handling

The Streamlit interface allows users to enter a movie title and view the resulting recommendations.

The application also handles common cases:

- **Invalid movie title:** Informs the user and provides available movie titles for selection.
- **No recommendations found:** Displays an appropriate message instead of presenting empty results as recommendations.
- **Empty input:** Prompts the user to enter or select a movie.
- **Database or API errors:** Handles failures and displays a user-facing error message where appropriate.

These checks make the application easier to test and use without introducing additional recommendation features.

### Application Screenshots

#### 1. Recommendation Result & AI Explanation
![Recommendation Output and AI Explanation](Screenshot%202026-10-14%20135049.png)

#### 2. Invalid Input & Available Movies Fallback
![Fallback and Available Movie Selection](Screenshot%202026-10-14%20135116.png)

## 9. Setup and Execution

### Prerequisites

- Python installed.
- A Neo4j AuraDB instance with the movie graph populated.
- A Google Gemini API key.

### Install dependencies

```bash
pip install neo4j python-dotenv google-genai streamlit
```

### Configure environment variables

Create a `.env` file in the project root:

```env
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=neo4j
GEMINI_API_KEY=your_gemini_api_key
```

Keep the actual credentials private. The `.env` file should not be committed to GitHub.

### Run the application

Start the Streamlit interface:

```bash
streamlit run streamlit_app.py
```

To run the core application through its command-line interface, use:

```bash
python app.py
```

## 10. Scope and Design Decisions

The implementation deliberately remains within the assignment requirements.

- Uses only `Movie`, `Actor`, and `Genre` nodes.
- Uses the required `ACTED_IN` and `IN_GENRE` relationships.
- Uses Cypher to find movies sharing actors or genres.
- Returns no more than three recommendations.
- Uses Gemini to explain recommendations based on retrieved graph evidence.
- Provides a simple Streamlit interface for interaction.

**No embeddings, vector database, GraphRAG, graph algorithms, machine-learning recommendation model, or additional similarity pipeline has been added.** The recommendations are obtained directly from the Neo4j graph using Cypher.

The small dataset and straightforward scoring approach keep the implementation focused on demonstrating graph modelling, querying, and LLM integration as required by the assignment.

## 11. Limitations

- Recommendations are limited to the movies and relationships present in the dataset.
- Movies without shared actors or genres are not recommended.
- Ranking is based on the number of shared actors and genres, without weights or personalization.
- Gemini explanations depend on the retrieved evidence and may still require verification.

These limitations are consistent with the deliberately simple, assignment-focused implementation.

## Conclusion

This project demonstrates how a graph database can be used to retrieve movie recommendations through relationships and how an LLM can explain those recommendations in natural language.

The implementation keeps graph retrieval, recommendation logic, explanation generation, and the user interface separate, while **strictly following the assignment scope and avoiding unnecessary complexity**.
