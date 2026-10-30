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
    build_cypher_query,
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

    def test_build_cypher_query_parameterization_safety(self):
        """Verify user values are NEVER concatenated into Cypher string."""
        from employee_interpreter import EmployeeQueryRequest, FilterCondition, FilterOperator, OperationType

        req = EmployeeQueryRequest(
            operation=OperationType.LIST,
            filters=[
                FilterCondition(field="location", operator=FilterOperator.EQUALS, value="Pune"),
                FilterCondition(field="salary", operator=FilterOperator.GREATER_THAN, value=1200000),
            ],
            limit=10,
        )
        cql, params = build_cypher_query(req)

        # Values must be bound as parameters, NOT in the raw string
        self.assertNotIn("'Pune'", cql)
        self.assertNotIn("1200000", cql)
        self.assertIn("$p0", cql)
        self.assertIn("$p1", cql)
        self.assertEqual(params["p0"], "Pune")
        self.assertEqual(params["p1"], 1200000)
        self.assertEqual(params["limit"], 10)

    def test_build_cypher_query_operators(self):
        """Verify comparison operators and between operator building."""
        from employee_interpreter import EmployeeQueryRequest, FilterCondition, FilterOperator, OperationType

        req = EmployeeQueryRequest(
            operation=OperationType.LIST,
            filters=[
                FilterCondition(field="salary", operator=FilterOperator.BETWEEN, value=[1200000, 2000000]),
                FilterCondition(field="location", operator=FilterOperator.NOT_EQUALS, value="Mumbai"),
                FilterCondition(field="age", operator=FilterOperator.LESS_THAN, value=35),
            ],
        )
        cql, params = build_cypher_query(req)

        self.assertIn("e.salary >= $p0 AND e.salary <= $p1", cql)
        self.assertIn("toLower(e.location) <> toLower($p2)", cql)
        self.assertIn("e.age < $p3", cql)
        self.assertEqual(params["p0"], 1200000)
        self.assertEqual(params["p1"], 2000000)
        self.assertEqual(params["p2"], "Mumbai")
        self.assertEqual(params["p3"], 35)

    def test_build_cypher_query_multi_skill(self):
        """Verify multi-skill matching matches distinct skills count."""
        from employee_interpreter import EmployeeQueryRequest, FilterCondition, FilterOperator, OperationType

        req = EmployeeQueryRequest(
            operation=OperationType.LIST,
            filters=[
                FilterCondition(field="skills", operator=FilterOperator.IN, value=["Python", "SQL"]),
            ],
        )
        cql, params = build_cypher_query(req)

        self.assertIn("UNWIND $req_skills AS req_skill", cql)
        self.assertIn("matched_skills = $req_skills_count", cql)
        self.assertEqual(params["req_skills"], ["python", "sql"])
        self.assertEqual(params["req_skills_count"], 2)

    def test_pune_headcount_bug_fixed(self):
        """Specifically verify 'How many employees have location as Pune?' counts Pune employees."""
        session = MagicMock()
        mock_rec = MockRecord({"count": 4})
        session.run.return_value.single.return_value = mock_rec

        res = handle_employee_query("How many employees have location as Pune?", session)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["query_type"], "count")
        self.assertEqual(res["count"], 4)
        self.assertIn("Pune", res["message"])

        # Check call args to verify parameterized location query was run
        call_args = session.run.call_args
        cql_executed = call_args[0][0]
        params_executed = call_args[1]

        self.assertIn("count(DISTINCT e)", cql_executed)
        self.assertIn("toLower(e.location) = toLower($p0)", cql_executed)
        self.assertEqual(params_executed["p0"], "Pune")

    def test_highest_headcount_department(self):
        """Verify 'Which department has the highest headcount?'."""
        session = MagicMock()
        mock_breakdown = [
            MockRecord({"department": "Engineering", "count": 12}),
            MockRecord({"department": "Data Science", "count": 6}),
        ]
        session.run.return_value.__iter__.return_value = mock_breakdown

        res = handle_employee_query("Which department has the highest headcount?", session)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["query_type"], "department_highest")
        self.assertEqual(res["data"][0]["department"], "Engineering")
        self.assertEqual(res["count"], 12)
        self.assertIn("Engineering", res["message"])

    def test_count_and_list_operation(self):
        """Verify 'How many Python developers are in Pune, and what are their names?'."""
        session = MagicMock()
        count_rec = MockRecord({"count": 1})
        emp1 = MockRecord({
            "employee_id": "EMP003",
            "name": "Aarav Patel",
            "designation": "Software Engineer",
            "department": "Engineering",
            "location": "Pune",
            "salary": 1800000,
            "skills": ["Python"],
        })

        # session.run is called first for count (single()), then for data (iteration)
        count_result = MagicMock()
        count_result.single.return_value = count_rec

        data_result = MagicMock()
        data_result.__iter__.return_value = [emp1]

        session.run.side_effect = [count_result, data_result]

        res = handle_employee_query("How many Python developers are in Pune, and what are their names?", session)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["query_type"], "count_and_list")
        self.assertEqual(res["count"], 1)
        self.assertEqual(len(res["data"]), 1)
        self.assertEqual(res["data"][0]["name"], "Aarav Patel")

    def test_get_field_skills(self):
        """Verify field retrieval like 'What skills does EMP001 have?'."""
        session = MagicMock()
        emp_record = MockRecord({
            "employee_id": "EMP001",
            "name": "Rajesh Sharma",
            "designation": "VP",
            "skills": ["Python", "Cloud Architecture"],
        })
        session.run.return_value.single.return_value = emp_record

        res = handle_employee_query("What skills does EMP001 have?", session)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["query_type"], "get_field")
        self.assertIn("Python, Cloud Architecture", res["message"])

    def test_clarification_on_empty_query(self):
        session = MagicMock()
        res = handle_employee_query("", session)
        self.assertEqual(res["status"], "empty")
        self.assertEqual(res["count"], 0)

    def test_dynamic_projection_single_field(self):
        """Query: 'Name all employees from Mumbai' -> projection RETURN e.name AS name."""
        from employee_interpreter import EmployeeQueryRequest, FilterCondition, FilterOperator, OperationType

        req = EmployeeQueryRequest(
            operation=OperationType.LIST,
            fields=["name"],
            filters=[FilterCondition(field="location", operator=FilterOperator.EQUALS, value="Mumbai")],
            limit=20,
        )
        cql, params = build_cypher_query(req)

        self.assertIn("RETURN e.name AS name", cql)
        self.assertNotIn("e.salary AS salary", cql)
        self.assertNotIn("collect(DISTINCT s.name)", cql)
        self.assertIn("toLower(e.location) = toLower($p0)", cql)
        self.assertIn("LIMIT $limit", cql)
        self.assertEqual(params["p0"], "Mumbai")
        self.assertEqual(params["limit"], 20)

    def test_dynamic_projection_multiple_fields(self):
        """Query: 'Show names and salaries of employees in Hyderabad' -> RETURN e.name AS name, e.salary AS salary."""
        from employee_interpreter import EmployeeQueryRequest, FilterCondition, FilterOperator, OperationType

        req = EmployeeQueryRequest(
            operation=OperationType.LIST,
            fields=["name", "salary"],
            filters=[FilterCondition(field="location", operator=FilterOperator.EQUALS, value="Hyderabad")],
            limit=15,
        )
        cql, params = build_cypher_query(req)

        self.assertIn("RETURN e.name AS name, e.salary AS salary", cql)
        self.assertNotIn("e.employee_id AS employee_id", cql)
        self.assertIn("LIMIT $limit", cql)
        self.assertEqual(params["limit"], 15)

    def test_dynamic_projection_with_department_and_manager(self):
        """Verify department and manager relations are included only when needed."""
        from employee_interpreter import EmployeeQueryRequest, FilterCondition, FilterOperator, OperationType

        req = EmployeeQueryRequest(
            operation=OperationType.LIST,
            fields=["name", "department", "manager_name"],
            filters=[],
            limit=10,
        )
        cql, params = build_cypher_query(req)

        self.assertIn("OPTIONAL MATCH (e)-[:WORKS_IN]->(d:Department)", cql)
        self.assertIn("OPTIONAL MATCH (e)-[:REPORTS_TO]->(m:Employee)", cql)
        self.assertIn("RETURN e.name AS name, d.name AS department, m.name AS manager_name", cql)
        self.assertIn("LIMIT $limit", cql)
        self.assertEqual(params["limit"], 10)

    def test_dynamic_projection_with_skills_collection(self):
        """Verify skill collection is matched and projected when skills field is requested."""
        from employee_interpreter import EmployeeQueryRequest, FilterCondition, FilterOperator, OperationType

        req = EmployeeQueryRequest(
            operation=OperationType.LIST,
            fields=["name", "skills"],
            filters=[],
            limit=10,
        )
        cql, params = build_cypher_query(req)

        self.assertIn("OPTIONAL MATCH (e)-[:HAS_SKILL]->(s:Skill)", cql)
        self.assertIn("collect(DISTINCT s.name) AS skills", cql)
        self.assertIn("RETURN e.name AS name, skills", cql)
        self.assertIn("LIMIT $limit", cql)

    def test_disallowed_field_injection_prevented(self):
        """Verify unsupported or malicious field names are not injected into Cypher projection."""
        from employee_interpreter import EmployeeQueryRequest, OperationType

        # Pass invalid field string bypassing Pydantic or via raw field list
        req = EmployeeQueryRequest(
            operation=OperationType.LIST,
            fields=["name"],
            limit=10,
        )
        # Force an arbitrary unvalidated field name
        req.fields = ["name", "malicious_field; DROP DATABASE--"]
        cql, params = build_cypher_query(req)

        self.assertNotIn("malicious_field", cql)
        self.assertNotIn("DROP DATABASE", cql)
        self.assertIn("RETURN e.name AS name", cql)

    def test_database_level_limit_and_accurate_count(self):
        """Verify that when 50 employees match and limit is 20:
        backend returns 20 records, reports count 50, and sets is_truncated to True.
        """
        from employee_interpreter import EmployeeQueryRequest, FilterCondition, FilterOperator, OperationType
        from employee_queries import execute_employee_query

        session = MagicMock()
        # Mock count query returning 50
        count_rec = MockRecord({"count": 50})
        count_result = MagicMock()
        count_result.single.return_value = count_rec

        # Mock data query returning 20 records (database-level LIMIT 20 applied)
        data_records = [
            MockRecord({"employee_id": f"EMP{i:03d}", "name": f"Employee {i}", "salary": 1000000 + i * 10000})
            for i in range(1, 21)
        ]
        data_result = MagicMock()
        data_result.__iter__.return_value = data_records

        session.run.side_effect = [count_result, data_result]

        req = EmployeeQueryRequest(
            operation=OperationType.LIST,
            fields=["employee_id", "name", "salary"],
            filters=[FilterCondition(field="location", operator=FilterOperator.EQUALS, value="Hyderabad")],
            limit=20,
        )

        res = execute_employee_query(req, session)

        # Check call arguments
        calls = session.run.call_args_list
        self.assertEqual(len(calls), 2)

        # First call is count query
        count_cql_call = calls[0][0][0]
        self.assertIn("count(DISTINCT e) AS count", count_cql_call)
        self.assertNotIn("LIMIT", count_cql_call)

        # Second call is data query
        data_cql_call = calls[1][0][0]
        data_params_call = calls[1][1]
        self.assertIn("LIMIT $limit", data_cql_call)
        self.assertEqual(data_params_call["limit"], 20)

        # Verify envelope response
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["count"], 50)
        self.assertEqual(len(res["data"]), 20)
        self.assertTrue(res["is_truncated"])
        self.assertIn("Displaying first 20 of 50", res["message"])
        self.assertEqual(res["fields"], ["employee_id", "name", "salary"])


if __name__ == "__main__":
    unittest.main()
