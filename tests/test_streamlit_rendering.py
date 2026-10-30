"""Unit tests for streamlit_app.py GET_FIELD rendering behavior."""

import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Mock streamlit and other optional external packages if not installed in current python env
if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = MagicMock()
if "neo4j" not in sys.modules:
    sys.modules["neo4j"] = MagicMock()
if "pandas" not in sys.modules:
    sys.modules["pandas"] = MagicMock()

import streamlit_app


class TestStreamlitGetFieldRendering(unittest.TestCase):

    def setUp(self):
        self.sample_emp = {
            "name": "Priya Sharma",
            "employee_id": "EMP012",
            "department": "Engineering",
            "designation": "Data Scientist",
            "location": "Pune",
            "salary": 1200000,
            "age": 28,
            "joining_date": "2021-06-15",
            "skills": ["Python", "SQL"],
            "manager_id": "EMP005",
            "manager_name": "Rajesh Kumar",
        }

    def test_format_field_value_salary(self):
        val = streamlit_app._format_field_value("salary", 1200000)
        self.assertEqual(val, streamlit_app._format_inr(1200000))

    def test_format_field_value_skills_list(self):
        val = streamlit_app._format_field_value("skills", ["Python", "SQL"])
        self.assertEqual(val, "Python, SQL")

    def test_format_field_value_none_or_missing(self):
        self.assertEqual(streamlit_app._format_field_value("location", None), "N/A")
        self.assertEqual(streamlit_app._format_field_value("skills", []), "None listed")

    @patch("streamlit_app.st")
    def test_render_get_field_salary_only(self, mock_st):
        formatted_sal = streamlit_app._format_inr(1200000)
        result = {
            "status": "success",
            "query_type": "get_field",
            "data": {"salary": 1200000},
            "employee": self.sample_emp,
            "count": 1,
            "message": f"Priya Sharma (EMP012) - Salary: {formatted_sal}",
        }

        with patch("streamlit_app.render_employee_profile") as mock_profile:
            streamlit_app.render_employee_results(result)
            mock_profile.assert_not_called()

        mock_st.success.assert_called_with(result["message"])
        mock_st.markdown.assert_called_once()
        html_arg = mock_st.markdown.call_args[0][0]

        # Verify Name & ID present
        self.assertIn("Priya Sharma", html_arg)
        self.assertIn("EMP012", html_arg)
        # Verify requested field present
        self.assertIn("Salary", html_arg)
        self.assertIn(formatted_sal, html_arg)
        # Verify unrequested fields are NOT present
        self.assertNotIn("Data Scientist", html_arg)
        self.assertNotIn("Engineering", html_arg)
        self.assertNotIn("Pune", html_arg)
        self.assertNotIn("Rajesh Kumar", html_arg)

    @patch("streamlit_app.st")
    def test_render_get_field_location_only(self, mock_st):
        formatted_sal = streamlit_app._format_inr(1200000)
        result = {
            "status": "success",
            "query_type": "get_field",
            "data": {"location": "Pune"},
            "employee": self.sample_emp,
            "count": 1,
            "message": "Priya Sharma (EMP012) - Location: Pune",
        }

        with patch("streamlit_app.render_employee_profile") as mock_profile:
            streamlit_app.render_employee_results(result)
            mock_profile.assert_not_called()

        html_arg = mock_st.markdown.call_args[0][0]
        self.assertIn("Location", html_arg)
        self.assertIn("Pune", html_arg)
        self.assertNotIn(formatted_sal, html_arg)
        self.assertNotIn("Python", html_arg)

    @patch("streamlit_app.st")
    def test_render_get_field_skills_only(self, mock_st):
        formatted_sal = streamlit_app._format_inr(1200000)
        result = {
            "status": "success",
            "query_type": "get_field",
            "data": {"skills": ["Python", "SQL"]},
            "employee": self.sample_emp,
            "count": 1,
            "message": "Priya Sharma (EMP012) - Skills: Python, SQL",
        }

        with patch("streamlit_app.render_employee_profile") as mock_profile:
            streamlit_app.render_employee_results(result)
            mock_profile.assert_not_called()

        html_arg = mock_st.markdown.call_args[0][0]
        self.assertIn("Skills", html_arg)
        self.assertIn("Python", html_arg)
        self.assertIn("SQL", html_arg)
        self.assertNotIn(formatted_sal, html_arg)
        self.assertNotIn("Pune", html_arg)

    @patch("streamlit_app.st")
    def test_render_get_field_multiple_fields(self, mock_st):
        formatted_sal = streamlit_app._format_inr(1200000)
        result = {
            "status": "success",
            "query_type": "get_field",
            "data": {"salary": 1200000, "designation": "Data Scientist"},
            "employee": self.sample_emp,
            "count": 1,
            "message": f"Priya Sharma (EMP012) - Salary: {formatted_sal} | Designation: Data Scientist",
        }

        with patch("streamlit_app.render_employee_profile") as mock_profile:
            streamlit_app.render_employee_results(result)
            mock_profile.assert_not_called()

        html_arg = mock_st.markdown.call_args[0][0]
        self.assertIn("Salary", html_arg)
        self.assertIn(formatted_sal, html_arg)
        self.assertIn("Designation", html_arg)
        self.assertIn("Data Scientist", html_arg)
        self.assertNotIn("Pune", html_arg)
        self.assertNotIn("Rajesh Kumar", html_arg)

    @patch("streamlit_app.st")
    def test_render_get_field_none_and_empty_values(self, mock_st):
        result = {
            "status": "success",
            "query_type": "get_field",
            "data": {"location": None, "skills": []},
            "employee": self.sample_emp,
            "count": 1,
            "message": "Priya Sharma (EMP012) - Location: None",
        }

        streamlit_app.render_employee_results(result)
        html_arg = mock_st.markdown.call_args[0][0]
        self.assertIn("Location", html_arg)
        self.assertIn("N/A", html_arg)
        self.assertIn("None listed", html_arg)

    @patch("streamlit_app.st")
    def test_existing_full_profile_rendering(self, mock_st):
        result = {
            "status": "success",
            "query_type": "employee_profile",
            "data": {
                "status": "exact",
                "employee": self.sample_emp,
            },
            "message": "Found employee record.",
        }

        with patch("streamlit_app.render_employee_profile") as mock_profile:
            streamlit_app.render_employee_results(result)
            mock_profile.assert_called_once_with(self.sample_emp)

    @patch("streamlit_app.st")
    def test_existing_multiple_match_candidate_rendering(self, mock_st):
        cand1 = dict(self.sample_emp, name="Priya 1")
        cand2 = dict(self.sample_emp, name="Priya 2")
        result = {
            "status": "success",
            "query_type": "employee_profile",
            "data": {
                "status": "multiple",
                "candidates": [cand1, cand2],
            },
            "message": "Multiple employees matched.",
        }

        with patch("streamlit_app.render_employee_profile") as mock_profile:
            streamlit_app.render_employee_results(result)
            self.assertEqual(mock_profile.call_count, 2)

    @patch("streamlit_app.st")
    def test_render_filtered_list_dynamic_fields_single(self, mock_st):
        """Verify dynamic field rendering for single field (e.g., 'Name all employees from Mumbai')."""
        result = {
            "status": "success",
            "query_type": "filtered_list",
            "data": [{"name": "Aarav Patel"}, {"name": "Meera Rao"}],
            "count": 2,
            "fields": ["name"],
            "message": "Found 2 employee(s) in Mumbai.",
        }

        with patch("streamlit_app.pd.DataFrame") as mock_df:
            streamlit_app.render_employee_results(result)
            mock_df.assert_called_once()
            table_rows = mock_df.call_args[0][0]
            # Should have only 'Name' column in table rows
            self.assertEqual(len(table_rows), 2)
            self.assertEqual(list(table_rows[0].keys()), ["Name"])
            self.assertEqual(table_rows[0]["Name"], "Aarav Patel")

    @patch("streamlit_app.st")
    def test_render_filtered_list_dynamic_fields_multi(self, mock_st):
        """Verify dynamic field rendering for multiple fields (name, salary)."""
        result = {
            "status": "success",
            "query_type": "filtered_list",
            "data": [
                {"name": "Aarav Patel", "salary": 1800000},
                {"name": "Sneha Reddy", "salary": 2200000},
            ],
            "count": 2,
            "fields": ["name", "salary"],
            "message": "Found 2 employee(s) in Hyderabad.",
        }

        with patch("streamlit_app.pd.DataFrame") as mock_df:
            streamlit_app.render_employee_results(result)
            mock_df.assert_called_once()
            table_rows = mock_df.call_args[0][0]
            self.assertEqual(len(table_rows), 2)
            self.assertEqual(list(table_rows[0].keys()), ["Name", "Salary"])
            self.assertEqual(table_rows[0]["Name"], "Aarav Patel")
            self.assertEqual(table_rows[0]["Salary"], streamlit_app._format_inr(1800000))

    @patch("streamlit_app.st")
    def test_render_count_and_list_dynamic_fields(self, mock_st):
        """Verify count_and_list metric and dynamic columns."""
        result = {
            "status": "success",
            "query_type": "count_and_list",
            "data": [{"name": "Aarav Patel", "skills": ["Python"]}],
            "count": 1,
            "fields": ["name", "skills"],
            "message": "Found 1 employee(s).",
        }

        with patch("streamlit_app.pd.DataFrame") as mock_df:
            streamlit_app.render_employee_results(result)
            mock_st.metric.assert_called_once_with(label="Total Matching Count", value=1)
            table_rows = mock_df.call_args[0][0]
            self.assertEqual(list(table_rows[0].keys()), ["Name", "Skills"])
            self.assertEqual(table_rows[0]["Skills"], "Python")


if __name__ == "__main__":
    unittest.main()
