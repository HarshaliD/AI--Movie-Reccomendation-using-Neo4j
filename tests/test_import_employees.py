"""Unit tests for Excel-to-Neo4j Ingestion Validation.

Tests validation rules:
- Valid dataset check
- Missing required columns
- Duplicate employee IDs
- Self-reporting check
- Invalid manager reference
- Circular reporting cycle detection
- Data type errors (negative salary, bad age, invalid date)
- Empty dataframe
Does not require Neo4j or network connectivity.
"""

from pathlib import Path
import sys
import unittest
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "movie-recommendation"))
sys.path.insert(0, str(ROOT_DIR))

from import_employees import validate_employees_data


class TestImportEmployeesValidation(unittest.TestCase):

    def setUp(self):
        self.valid_data = [
            {
                "employee_id": "EMP001",
                "name": "Rajesh Sharma",
                "age": 50,
                "department": "Engineering",
                "designation": "VP of Technology",
                "salary": 4500000,
                "location": "Bengaluru",
                "joining_date": "2019-01-15",
                "manager_id": "",
                "skills": "Python, Cloud, Leadership",
            },
            {
                "employee_id": "EMP002",
                "name": "Priya Sundaram",
                "age": 40,
                "department": "Engineering",
                "designation": "Engineering Manager",
                "salary": 3200000,
                "location": "Bengaluru",
                "joining_date": "2020-03-01",
                "manager_id": "EMP001",
                "skills": "Python, Distributed Systems",
            },
            {
                "employee_id": "EMP003",
                "name": "Aarav Patel",
                "age": 30,
                "department": "Engineering",
                "designation": "Software Engineer",
                "salary": 1800000,
                "location": "Hyderabad",
                "joining_date": "2021-06-15",
                "manager_id": "EMP002",
                "skills": "Python, SQL, Docker",
            },
        ]

    def test_valid_dataset(self):
        df = pd.DataFrame(self.valid_data)
        is_valid, errors, cleaned = validate_employees_data(df)
        self.assertTrue(is_valid, f"Expected valid data, but got errors: {errors}")
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(cleaned), 3)

    def test_missing_required_columns(self):
        data = [r.copy() for r in self.valid_data]
        df = pd.DataFrame(data).drop(columns=["salary", "skills"])
        is_valid, errors, cleaned = validate_employees_data(df)
        self.assertFalse(is_valid)
        self.assertTrue(any("Missing required columns" in e for e in errors))

    def test_duplicate_employee_ids(self):
        data = [r.copy() for r in self.valid_data]
        data.append({
            "employee_id": "EMP001",  # duplicate ID
            "name": "Another Person",
            "age": 28,
            "department": "Finance",
            "designation": "Analyst",
            "salary": 1200000,
            "location": "Pune",
            "joining_date": "2022-01-01",
            "manager_id": "",
            "skills": "Excel, SQL",
        })
        df = pd.DataFrame(data)
        is_valid, errors, cleaned = validate_employees_data(df)
        self.assertFalse(is_valid)
        self.assertTrue(any("Duplicate employee_id 'EMP001'" in e for e in errors))

    def test_self_reporting(self):
        data = [r.copy() for r in self.valid_data]
        data[0]["manager_id"] = "EMP001"  # EMP001 reports to EMP001
        df = pd.DataFrame(data)
        is_valid, errors, cleaned = validate_employees_data(df)
        self.assertFalse(is_valid)
        self.assertTrue(any("Self-reporting detected" in e for e in errors))

    def test_invalid_manager_reference(self):
        data = [r.copy() for r in self.valid_data]
        data[1]["manager_id"] = "EMP999"  # EMP999 does not exist
        df = pd.DataFrame(data)
        is_valid, errors, cleaned = validate_employees_data(df)
        self.assertFalse(is_valid)
        self.assertTrue(any("Referenced manager_id 'EMP999' does not exist" in e for e in errors))

    def test_circular_reporting(self):
        data = [
            {
                "employee_id": "EMP001",
                "name": "Person A",
                "age": 40,
                "department": "Engineering",
                "designation": "Lead",
                "salary": 2000000,
                "location": "Pune",
                "joining_date": "2020-01-01",
                "manager_id": "EMP002",  # A reports to B
                "skills": "Python",
            },
            {
                "employee_id": "EMP002",
                "name": "Person B",
                "age": 35,
                "department": "Engineering",
                "designation": "Lead",
                "salary": 2000000,
                "location": "Pune",
                "joining_date": "2020-01-01",
                "manager_id": "EMP003",  # B reports to C
                "skills": "Python",
            },
            {
                "employee_id": "EMP003",
                "name": "Person C",
                "age": 32,
                "department": "Engineering",
                "designation": "Lead",
                "salary": 2000000,
                "location": "Pune",
                "joining_date": "2020-01-01",
                "manager_id": "EMP001",  # C reports to A -> cycle!
                "skills": "Python",
            },
        ]
        df = pd.DataFrame(data)
        is_valid, errors, cleaned = validate_employees_data(df)
        self.assertFalse(is_valid)
        self.assertTrue(any("Circular reporting relationship detected" in e for e in errors))

    def test_invalid_numeric_types(self):
        data = [r.copy() for r in self.valid_data]
        data[0]["age"] = "invalid_age"
        data[1]["salary"] = -50000
        df = pd.DataFrame(data)
        is_valid, errors, cleaned = validate_employees_data(df)
        self.assertFalse(is_valid)
        self.assertTrue(any("age' must be a valid integer" in e for e in errors))
        self.assertTrue(any("salary' must be positive" in e for e in errors))

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=[
            "employee_id", "name", "age", "department", "designation", "salary", "location", "joining_date", "skills", "manager_id"
        ])
        is_valid, errors, cleaned = validate_employees_data(df)
        self.assertFalse(is_valid)
        self.assertTrue(any("empty" in e.lower() for e in errors))


if __name__ == "__main__":
    unittest.main()
