"""Unit tests for Movie Recommendations and Gemini explanation.

Tests:
- Movie existence check (found, not found)
- Recommendation retrieval and query structure
- Available movies listing
- Gemini explanation generation with valid client
- Error handling when Gemini explanation fails
Uses mocks so no live database or internet access is required.
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "movie-recommendation"))
sys.path.insert(0, str(ROOT_DIR))

from app import (
    check_movie_exists,
    explain_recommendation,
    get_available_movies,
    get_recommendations,
)


class TestMovieRecommendationsOffline(unittest.TestCase):

    def test_check_movie_exists_true(self):
        session = MagicMock()
        session.run.return_value.single.return_value = {"title": "3 Idiots"}
        self.assertTrue(check_movie_exists(session, "3 Idiots"))

    def test_check_movie_exists_false(self):
        session = MagicMock()
        session.run.return_value.single.return_value = None
        self.assertFalse(check_movie_exists(session, "Unknown Movie"))

    def test_get_available_movies(self):
        session = MagicMock()
        session.run.return_value.__iter__.return_value = [
            {"title": "3 Idiots"},
            {"title": "Dangal"},
            {"title": "PK"},
        ]
        movies = get_available_movies(session)
        self.assertEqual(movies, ["3 Idiots", "Dangal", "PK"])

    def test_get_recommendations(self):
        session = MagicMock()
        rec_data = [
            MagicMock(data=lambda: {
                "movie": "PK",
                "sharedActors": ["Aamir Khan"],
                "sharedGenres": ["Comedy", "Drama"],
            }),
            MagicMock(data=lambda: {
                "movie": "Dangal",
                "sharedActors": ["Aamir Khan"],
                "sharedGenres": ["Drama"],
            }),
        ]
        session.run.return_value.__iter__.return_value = rec_data

        recs = get_recommendations(session, "3 Idiots")
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0]["movie"], "PK")
        self.assertIn("Aamir Khan", recs[0]["sharedActors"])

    def test_explain_recommendation_success(self):
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Both movies star Aamir Khan and blend emotional drama with lighthearted comedy."
        client.models.generate_content.return_value = mock_response

        recommendation = {
            "movie": "PK",
            "sharedActors": ["Aamir Khan"],
            "sharedGenres": ["Comedy", "Drama"],
        }
        explanation = explain_recommendation("3 Idiots", recommendation, client=client)
        self.assertEqual(explanation, mock_response.text)

    def test_explain_recommendation_empty_response_raises(self):
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = ""
        client.models.generate_content.return_value = mock_response

        recommendation = {
            "movie": "PK",
            "sharedActors": ["Aamir Khan"],
            "sharedGenres": ["Comedy"],
        }
        with self.assertRaises(RuntimeError):
            explain_recommendation("3 Idiots", recommendation, client=client)

    def test_explain_recommendation_none_client_raises(self):
        recommendation = {
            "movie": "PK",
            "sharedActors": ["Aamir Khan"],
            "sharedGenres": ["Comedy"],
        }
        with self.assertRaises(ValueError):
            explain_recommendation("3 Idiots", recommendation, client=None)


if __name__ == "__main__":
    unittest.main()
