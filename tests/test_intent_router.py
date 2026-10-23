"""Unit tests for Intent Router.

Tests classification into:
- movie
- employee
- general
- ambiguous
Does not require Neo4j or network connectivity.
"""

from pathlib import Path
import sys
import unittest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "movie-recommendation"))
sys.path.insert(0, str(ROOT_DIR))

from intent_router import route_intent


class TestIntentRouter(unittest.TestCase):

    def test_movie_questions(self):
        queries = [
            "Recommend movies like 3 Idiots",
            "Show movies featuring Aamir Khan",
            "Tell me about films starring Kareena Kapoor",
            "Suggest cinema similar to Chhichhore",
            "3 Idiots",
            "Dangal",
            "What movie should I watch next?",
        ]
        for q in queries:
            with self.subTest(query=q):
                res = route_intent(q)
                self.assertEqual(res.intent, "movie", f"Query '{q}' failed with {res.intent}: {res.reason}")

    def test_employee_questions(self):
        queries = [
            "Who works in Data Science?",
            "Who is EMP012's manager?",
            "Which employees know Python and SQL?",
            "List Hyderabad employees earning above 12 lakh",
            "How many employees work in Engineering?",
            "Who reports to EMP001?",
            "Show employees with designation Senior Software Engineer",
            "Find staff in Bengaluru office",
        ]
        for q in queries:
            with self.subTest(query=q):
                res = route_intent(q)
                self.assertEqual(res.intent, "employee", f"Query '{q}' failed with {res.intent}: {res.reason}")

    def test_greetings_and_courtesy(self):
        queries = [
            "Hi",
            "Hello",
            "Hey",
            "Good morning",
            "Namaste",
            "thanks",
            "thank you so much",
            "bye",
            "goodbye",
            "Help me",
            "what can you do",
        ]
        for q in queries:
            with self.subTest(query=q):
                res = route_intent(q)
                self.assertEqual(res.intent, "general", f"Query '{q}' failed with {res.intent}: {res.reason}")

    def test_empty_input(self):
        for empty_val in ["", "   ", "\t\n", None]:
            with self.subTest(val=empty_val):
                res = route_intent(empty_val)
                self.assertEqual(res.intent, "general")

    def test_ambiguous_questions(self):
        queries = [
            "Show me the best ones",
            "find them",
            "show all",
            "list all",
            "tell me more",
            "give me options",
        ]
        for q in queries:
            with self.subTest(query=q):
                res = route_intent(q)
                self.assertEqual(res.intent, "ambiguous", f"Query '{q}' should be ambiguous but got {res.intent}")

    def test_mixed_domain_questions(self):
        queries = [
            "Recommend movies for employees",
            "Show movies about engineering managers",
            "Employees acting in films",
        ]
        for q in queries:
            with self.subTest(query=q):
                res = route_intent(q)
                self.assertEqual(res.intent, "ambiguous", f"Query '{q}' should be ambiguous but got {res.intent}")

    def test_unrecognized_input_not_movie_fallback(self):
        queries = [
            "Random gibberish xyz987",
            "completely unknown sentence with no tokens",
        ]
        for q in queries:
            with self.subTest(query=q):
                res = route_intent(q)
                self.assertNotEqual(res.intent, "movie", "Unrecognized input must never fall through to movie handler!")
                self.assertEqual(res.intent, "ambiguous")


if __name__ == "__main__":
    unittest.main()
