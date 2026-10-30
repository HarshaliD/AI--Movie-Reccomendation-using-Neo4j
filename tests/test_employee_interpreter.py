"""Unit tests for Employee Natural Language Query Interpreter.

Tests:
- Pydantic schema validation (types, operators, bounds, aliases)
- Unsupported fields & operator validation
- Indian salary parsing and normalizations (lakh, LPA, crore, INR numeric)
- Safe LLM output JSON extraction
- Gemini API response parsing with mocks
- Fallback to deterministic parser on LLM error/malformed JSON
- Deterministic fallback parser coverage for all standard query patterns
"""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "movie-recommendation"))
sys.path.insert(0, str(ROOT_DIR))

from employee_interpreter import (
    EmployeeQueryRequest,
    EntityType,
    FilterCondition,
    FilterOperator,
    OperationType,
    SortSpec,
    deterministic_fallback_parser,
    extract_json_from_llm_response,
    interpret_employee_query,
    parse_indian_salary,
)


class TestEmployeeInterpreter(unittest.TestCase):

    def test_parse_indian_salary(self):
        # Lakh and LPA formats
        self.assertEqual(parse_indian_salary("12 lakh"), 1200000)
        self.assertEqual(parse_indian_salary("12 lakhs"), 1200000)
        self.assertEqual(parse_indian_salary("12 LPA"), 1200000)
        self.assertEqual(parse_indian_salary("12.5 lakh"), 1250000)
        self.assertEqual(parse_indian_salary("12L"), 1200000)
        self.assertEqual(parse_indian_salary("₹12,00,000"), 1200000)
        self.assertEqual(parse_indian_salary(1200000), 1200000)

        # Crore formats
        self.assertEqual(parse_indian_salary("1.5 crore"), 15000000)
        self.assertEqual(parse_indian_salary("1.5 cr"), 15000000)
        self.assertEqual(parse_indian_salary("2 crore"), 20000000)

        # None and invalid
        self.assertIsNone(parse_indian_salary(None))
        self.assertIsNone(parse_indian_salary("not a number"))

    def test_filter_condition_validation_and_aliases(self):
        # Valid location
        f = FilterCondition(field="location", operator=FilterOperator.EQUALS, value="Pune")
        self.assertEqual(f.field, "location")
        self.assertEqual(f.value, "Pune")

        # Alias: dept -> department
        f_dept = FilterCondition(field="dept", operator=FilterOperator.EQUALS, value="Engineering")
        self.assertEqual(f_dept.field, "department")

        # Alias: pay -> salary with conversion
        f_sal = FilterCondition(field="pay", operator=FilterOperator.GREATER_THAN, value="15 lakh")
        self.assertEqual(f_sal.field, "salary")
        self.assertEqual(f_sal.value, 1500000)

        # Salary with between
        f_between = FilterCondition(
            field="salary",
            operator=FilterOperator.BETWEEN,
            value=["12 lakh", "20 lakh"],
        )
        self.assertEqual(f_between.value, [1200000, 2000000])

        # Unsupported field rejection
        with self.assertRaises(ValueError):
            FilterCondition(field="favorite_food", operator=FilterOperator.EQUALS, value="Pizza")

    def test_sort_spec_validation(self):
        # Valid sort
        s = SortSpec(field="salary", direction="desc")
        self.assertEqual(s.field, "salary")
        self.assertEqual(s.direction, "desc")

        # Alias: pay -> salary
        s_alias = SortSpec(field="pay", direction="asc")
        self.assertEqual(s_alias.field, "salary")

        # Unsupported sort field rejection
        with self.assertRaises(ValueError):
            SortSpec(field="secret_code", direction="asc")

    def test_employee_query_request_bounds(self):
        # Valid request
        req = EmployeeQueryRequest(
            operation=OperationType.LIST,
            limit=50,
        )
        self.assertEqual(req.limit, 50)

        # Limit out of bounds (< 1)
        with self.assertRaises(ValueError):
            EmployeeQueryRequest(limit=0)

        # Limit out of bounds (> 100)
        with self.assertRaises(ValueError):
            EmployeeQueryRequest(limit=150)

    def test_extract_json_from_llm_response(self):
        # Code fence format
        fenced = "```json\n{\"operation\": \"count\", \"entity\": \"employee\"}\n```"
        res = extract_json_from_llm_response(fenced)
        self.assertEqual(res["operation"], "count")

        # Fenced without json tag
        fenced_plain = "```\n{\"operation\": \"list\"}\n```"
        res2 = extract_json_from_llm_response(fenced_plain)
        self.assertEqual(res2["operation"], "list")

        # Embedded in text
        text_wrapped = "Here is your JSON:\n{\"operation\": \"profile\", \"target_identifier\": \"EMP001\"}\nHope this helps!"
        res3 = extract_json_from_llm_response(text_wrapped)
        self.assertEqual(res3["target_identifier"], "EMP001")

        # Malformed
        with self.assertRaises(Exception):
            extract_json_from_llm_response("This contains no JSON object")

    def test_interpret_with_mock_gemini_success(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "entity": "employee",
            "operation": "list",
            "filters": [
                {"field": "location", "operator": "=", "value": "Pune"},
                {"field": "salary", "operator": ">", "value": 1200000},
            ],
            "combine_with": "AND",
            "limit": 10,
        })
        mock_client.models.generate_content.return_value = mock_response

        req = interpret_employee_query("Find Pune employees earning above 12 lakh", client=mock_client)
        self.assertEqual(req.operation, OperationType.LIST)
        self.assertEqual(len(req.filters), 2)
        self.assertEqual(req.filters[0].value, "Pune")
        self.assertEqual(req.filters[1].value, 1200000)
        self.assertEqual(req.filters[1].operator, FilterOperator.GREATER_THAN)

    def test_interpret_gemini_failure_falls_back_to_deterministic(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API Rate Limit or Connection Error")

        # Should not raise exception; falls back to deterministic parser
        req = interpret_employee_query("How many employees are in Pune?", client=mock_client)
        self.assertEqual(req.operation, OperationType.COUNT)
        self.assertEqual(len(req.filters), 1)
        self.assertEqual(req.filters[0].field, "location")
        self.assertEqual(req.filters[0].value, "Pune")

    def test_deterministic_fallback_pune_count_bug(self):
        # Fixes the bug: "How many employees have location as Pune?"
        req = deterministic_fallback_parser("How many employees have location as Pune?")
        self.assertEqual(req.operation, OperationType.COUNT)
        self.assertEqual(len(req.filters), 1)
        self.assertEqual(req.filters[0].field, "location")
        self.assertEqual(req.filters[0].value, "Pune")
        self.assertEqual(req.filters[0].operator, FilterOperator.EQUALS)

        # "How many employees are in Pune?"
        req2 = deterministic_fallback_parser("How many employees are in Pune?")
        self.assertEqual(req2.operation, OperationType.COUNT)
        self.assertEqual(req2.filters[0].value, "Pune")

    def test_deterministic_fallback_department_queries(self):
        # "List employees in Engineering"
        req = deterministic_fallback_parser("List employees in Engineering")
        self.assertEqual(req.operation, OperationType.LIST)
        self.assertEqual(req.filters[0].field, "department")
        self.assertEqual(req.filters[0].value, "Engineering")

        # "How many employees work in each department?"
        req_each = deterministic_fallback_parser("How many employees work in each department?")
        self.assertEqual(req_each.operation, OperationType.AGGREGATE_DEPARTMENT)

        # "Which department has the highest headcount?"
        req_top = deterministic_fallback_parser("Which department has the highest headcount?")
        self.assertEqual(req_top.operation, OperationType.AGGREGATE_DEPARTMENT)
        self.assertEqual(req_top.sort.field, "count")
        self.assertEqual(req_top.sort.direction, "desc")
        self.assertEqual(req_top.limit, 1)

    def test_deterministic_fallback_salary_queries(self):
        # Greater than 12 lakh (strictly >)
        req_gt = deterministic_fallback_parser("Find employees earning more than 12 lakh")
        self.assertEqual(req_gt.filters[0].field, "salary")
        self.assertEqual(req_gt.filters[0].operator, FilterOperator.GREATER_THAN)
        self.assertEqual(req_gt.filters[0].value, 1200000)

        # At least 12 lakh (>=)
        req_gte = deterministic_fallback_parser("Find employees earning at least 12 lakh")
        self.assertEqual(req_gte.filters[0].operator, FilterOperator.GREATER_THAN_OR_EQUAL)
        self.assertEqual(req_gte.filters[0].value, 1200000)

        # Between 12 lakh and 20 lakh
        req_between = deterministic_fallback_parser("Find employees earning between 12 lakh and 20 lakh")
        self.assertEqual(req_between.filters[0].operator, FilterOperator.BETWEEN)
        self.assertEqual(req_between.filters[0].value, [1200000, 2000000])

        # Five highest-paid employees
        req_top5 = deterministic_fallback_parser("List the five highest-paid employees")
        self.assertEqual(req_top5.sort.field, "salary")
        self.assertEqual(req_top5.sort.direction, "desc")
        self.assertEqual(req_top5.limit, 5)

    def test_deterministic_fallback_skills_queries(self):
        # Python and SQL
        req = deterministic_fallback_parser("Find employees with both Python and SQL skills")
        self.assertEqual(req.filters[0].field, "skills")
        self.assertIn("Python", req.filters[0].value)
        self.assertIn("SQL", req.filters[0].value)

    def test_deterministic_fallback_location_or_negation(self):
        # Pune or Hyderabad
        req_or = deterministic_fallback_parser("Find employees in Pune or Hyderabad")
        self.assertEqual(req_or.filters[0].field, "location")
        self.assertEqual(req_or.filters[0].operator, FilterOperator.IN)
        self.assertIn("Pune", req_or.filters[0].value)
        self.assertIn("Hyderabad", req_or.filters[0].value)

        # Not in Pune
        req_not = deterministic_fallback_parser("Find employees who are not in Pune")
        self.assertEqual(req_not.filters[0].field, "location")
        self.assertEqual(req_not.filters[0].operator, FilterOperator.NOT_EQUALS)
        self.assertEqual(req_not.filters[0].value, "Pune")

    def test_deterministic_fallback_age(self):
        req = deterministic_fallback_parser("Find employees aged below 30")
        self.assertEqual(req.filters[0].field, "age")
        self.assertEqual(req.filters[0].operator, FilterOperator.LESS_THAN)
        self.assertEqual(req.filters[0].value, 30)

    def test_deterministic_fallback_manager_and_reports(self):
        # Manager
        req_mgr = deterministic_fallback_parser("Who manages Priya?")
        self.assertEqual(req_mgr.operation, OperationType.MANAGER_LOOKUP)
        self.assertEqual(req_mgr.target_identifier, "Priya")

        # Direct reports
        req_rep = deterministic_fallback_parser("Who reports directly to Rahul?")
        self.assertEqual(req_rep.operation, OperationType.DIRECT_REPORTS)
        self.assertEqual(req_rep.target_identifier, "Rahul")

    def test_deterministic_fallback_profile_and_field(self):
        # Tell me about EMP001
        req_prof = deterministic_fallback_parser("Tell me about EMP001")
        self.assertEqual(req_prof.operation, OperationType.PROFILE)
        self.assertEqual(req_prof.target_identifier, "EMP001")

        # What skills does EMP001 have?
        req_field = deterministic_fallback_parser("What skills does EMP001 have?")
        self.assertEqual(req_field.operation, OperationType.GET_FIELD)
        self.assertEqual(req_field.target_identifier, "EMP001")
        self.assertEqual(req_field.fields, ["skills"])

    def test_deterministic_fallback_count_and_list(self):
        req = deterministic_fallback_parser("How many Python developers are in Pune, and what are their names?")
        self.assertEqual(req.operation, OperationType.COUNT_AND_LIST)
        self.assertTrue(req.include_count)


if __name__ == "__main__":
    unittest.main()
