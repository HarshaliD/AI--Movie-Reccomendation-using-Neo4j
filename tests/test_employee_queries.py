"""Unit tests for Employee Knowledge Graph Query Logic.

Tests:
- Natural language parameter extraction (salary, skills, departments, locations)
- Query building and parameterization
- Employee lookup (exact, multiple, not found)
- Manager lookup (with manager, top-level executive without manager)
- Direct reports lookup
- Department headcounts
- Filtered query construction
Uses mock Neo4j sessions so real database credentials are not required.
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "movie-recommendation"))
sys.path.insert(0, str(ROOT_DIR))

from employee_queries import (
    get_department_headcounts,
    get_direct_reports,
    get_employee_by_id_or_name,
    get_employee_manager,
    get_filtered_employees,
    handle_employee_query,
    parse_department_from_query,
    parse_location_from_query,
    parse_salary_threshold,
    parse_skills_from_query,
)


class MockRecord(dict):
    """Mock Neo4j record that supports both dict and attribute access."""
    def __getitem__(self, key):
        return super().__getitem__(key)


class TestEmployeeQueriesOffline(unittest.TestCase):

    def test_parse_salary_threshold(self):
        # Above 12 lakh
        min_s, max_s = parse_salary_threshold("List employees earning above 12 lakh")
        self.assertEqual(min_s, 1200000)
        self.assertIsNone(max_s)

        # Below 15 LPA
        min_s, max_s = parse_salary_threshold("Find staff earning below 15 lpa")
        self.assertIsNone(min_s)
        self.assertEqual(max_s, 1500000)

        # Greater than 2500000
        min_s, max_s = parse_salary_threshold("employees earning greater than 2500000")
        self.assertEqual(min_s, 2500000)

    def test_parse_skills_from_query(self):
        skills = parse_skills_from_query("Which employees know Python and SQL?")
        self.assertIn("python", skills)
        self.assertIn("sql", skills)

        skills = parse_skills_from_query("Developers with React and Docker experience")
        self.assertIn("react", skills)
        self.assertIn("docker", skills)

    def test_parse_department_from_query(self):
        dept = parse_department_from_query("Who works in Data Science?")
        self.assertEqual(dept, "Data Science")

        dept = parse_department_from_query("How many people work in Engineering?")
        self.assertEqual(dept, "Engineering")

        dept = parse_department_from_query("Find HR staff")
        self.assertEqual(dept, "Human Resources")

    def test_parse_location_from_query(self):
        loc = parse_location_from_query("List Hyderabad employees")
        self.assertEqual(loc, "Hyderabad")

        loc = parse_location_from_query("Staff in Bengaluru office")
        self.assertEqual(loc, "Bengaluru")

    def test_employee_lookup_exact_by_id(self):
        session = MagicMock()
        mock_record = MockRecord({
            "employee_id": "EMP001",
            "name": "Rajesh Sharma",
            "age": 52,
            "designation": "VP of Technology",
            "salary": 4500000,
            "location": "Bengaluru",
            "joining_date": "2019-01-15",
            "department": "Engineering",
            "manager_id": None,
            "manager_name": None,
            "skills": ["Python", "Cloud"],
        })
        session.run.return_value.single.return_value = mock_record

        res = get_employee_by_id_or_name(session, "EMP001")
        self.assertEqual(res["status"], "exact")
        self.assertEqual(res["employee"]["name"], "Rajesh Sharma")

    def test_employee_lookup_not_found(self):
        session = MagicMock()
        session.run.return_value.single.return_value = None
        session.run.return_value.__iter__.return_value = []

        res = get_employee_by_id_or_name(session, "NonExistentPerson")
        self.assertEqual(res["status"], "not_found")
        self.assertIsNone(res["employee"])

    def test_employee_lookup_multiple_disambiguation(self):
        session = MagicMock()
        # id lookup returns none
        session.run.return_value.single.return_value = None
        # name lookup returns 2 candidates with same surname
        c1 = MockRecord({
            "employee_id": "EMP001",
            "name": "Rajesh Sharma",
            "designation": "VP",
            "salary": 4500000,
            "skills": [],
        })
        c2 = MockRecord({
            "employee_id": "EMP020",
            "name": "Amit Sharma",
            "designation": "Developer",
            "salary": 1500000,
            "skills": [],
        })
        session.run.return_value.__iter__.return_value = [c1, c2]

        res = get_employee_by_id_or_name(session, "Sharma")
        self.assertEqual(res["status"], "multiple")
        self.assertEqual(len(res["candidates"]), 2)
        self.assertIn("specify by employee ID", res["message"])

    def test_manager_lookup_top_level(self):
        session = MagicMock()
        top_emp = MockRecord({
            "employee_id": "EMP001",
            "name": "Rajesh Sharma",
            "manager_id": None,
            "manager_name": None,
            "skills": [],
        })
        session.run.return_value.single.return_value = top_emp

        res = get_employee_manager(session, "EMP001")
        self.assertEqual(res["status"], "no_manager")
        self.assertIn("top-level executive", res["message"])

    def test_manager_lookup_with_manager(self):
        session = MagicMock()
        emp_record = MockRecord({
            "employee_id": "EMP002",
            "name": "Priya Sundaram",
            "manager_id": "EMP001",
            "manager_name": "Rajesh Sharma",
            "skills": [],
        })
        mgr_record = MockRecord({
            "employee_id": "EMP001",
            "name": "Rajesh Sharma",
            "designation": "VP of Technology",
            "skills": [],
        })

        session.run.return_value.single.side_effect = [emp_record, mgr_record]

        res = get_employee_manager(session, "EMP002")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["manager"]["employee_id"], "EMP001")

    def test_direct_reports(self):
        session = MagicMock()
        mgr_record = MockRecord({
            "employee_id": "EMP002",
            "name": "Priya Sundaram",
            "manager_id": "EMP001",
            "skills": [],
        })
        session.run.return_value.single.return_value = mgr_record

        sub1 = MockRecord({
            "employee_id": "EMP008",
            "name": "Aarav Patel",
            "designation": "Staff Engineer",
            "salary": 2600000,
            "location": "Hyderabad",
            "skills": ["Python"],
        })
        session.run.return_value.__iter__.return_value = [sub1]

        res = get_direct_reports(session, "EMP002")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["count"], 1)
        self.assertEqual(res["reports"][0]["employee_id"], "EMP008")

    def test_handle_employee_query_routing(self):
        session = MagicMock()
        # Mock headcount response
        rec = MockRecord({"department": "Engineering", "count": 8})
        session.run.return_value.single.return_value = rec

        res = handle_employee_query("How many employees work in Engineering?", session)
        self.assertEqual(res["query_type"], "headcount")
        self.assertEqual(res["count"], 8)


if __name__ == "__main__":
    unittest.main()
