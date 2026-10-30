"""Employee Knowledge Graph Query Handler.

Executes strictly parameterized, allowlisted Cypher queries against Neo4j AuraDB based on
validated EmployeeQueryRequest structures. Ensures zero Cypher injection, avoids cartesian
product duplicate records, and supports flexible operations (count, list, profile, get_field,
manager lookup, direct reports, department aggregations, count_and_list).
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from employee_interpreter import (
    ALLOWED_FILTER_FIELDS,
    ALLOWED_SORT_FIELDS,
    EmployeeQueryRequest,
    EntityType,
    FilterCondition,
    FilterOperator,
    OperationType,
    SortSpec,
    interpret_employee_query,
)

MAX_RESULTS_LIMIT = 20

# Safe Property Mapping for Cypher building
CYPHER_PROPERTY_MAP = {
    "employee_id": "e.employee_id",
    "name": "e.name",
    "age": "e.age",
    "designation": "e.designation",
    "salary": "e.salary",
    "location": "e.location",
    "joining_date": "e.joining_date",
    "department": "d.name",
    "manager_id": "m.employee_id",
    "manager_name": "m.name",
}

DEFAULT_LIST_FIELDS = [
    "employee_id",
    "name",
    "designation",
    "department",
    "location",
    "salary",
    "skills",
]

# Projection expressions for dynamic RETURN clause
FIELD_EXPRESSION_MAP = {
    "employee_id": "e.employee_id AS employee_id",
    "name": "e.name AS name",
    "age": "e.age AS age",
    "designation": "e.designation AS designation",
    "salary": "e.salary AS salary",
    "location": "e.location AS location",
    "joining_date": "e.joining_date AS joining_date",
    "department": "d.name AS department",
    "manager_id": "m.employee_id AS manager_id",
    "manager_name": "m.name AS manager_name",
    "skills": "skills",
}


def _format_inr(amount: Optional[Union[int, float]]) -> str:
    """Format an integer amount into an INR currency string."""
    if amount is None:
        return "N/A"
    return f"₹{amount:,.0f}"


def get_employee_by_id_or_name(session, identifier: str) -> Dict[str, Any]:
    """Lookup employee by employee_id or partial/full name.

    Returns:
        dict: {
            "status": "exact" | "multiple" | "not_found",
            "employee": dict | None,
            "candidates": list[dict],
            "message": str
        }
    """
    clean_id = identifier.strip().upper()
    clean_term = identifier.strip()

    # 1. Try exact employee_id lookup
    cql_id = """
    MATCH (e:Employee)
    WHERE toUpper(e.employee_id) = $emp_id
    OPTIONAL MATCH (e)-[:WORKS_IN]->(d:Department)
    OPTIONAL MATCH (e)-[:REPORTS_TO]->(m:Employee)
    OPTIONAL MATCH (e)-[:HAS_SKILL]->(s:Skill)
    RETURN e.employee_id AS employee_id,
           e.name AS name,
           e.age AS age,
           e.designation AS designation,
           e.salary AS salary,
           e.location AS location,
           e.joining_date AS joining_date,
           d.name AS department,
           m.employee_id AS manager_id,
           m.name AS manager_name,
           collect(DISTINCT s.name) AS skills
    """
    record = session.run(cql_id, emp_id=clean_id).single()
    if record and record["employee_id"]:
        return {
            "status": "exact",
            "employee": dict(record),
            "candidates": [],
            "message": f"Found employee record for {record['name']} ({record['employee_id']}).",
        }

    # 2. Try name search (case-insensitive contains)
    cql_name = """
    MATCH (e:Employee)
    WHERE toLower(e.name) CONTAINS toLower($name)
    OPTIONAL MATCH (e)-[:WORKS_IN]->(d:Department)
    OPTIONAL MATCH (e)-[:REPORTS_TO]->(m:Employee)
    OPTIONAL MATCH (e)-[:HAS_SKILL]->(s:Skill)
    RETURN e.employee_id AS employee_id,
           e.name AS name,
           e.age AS age,
           e.designation AS designation,
           e.salary AS salary,
           e.location AS location,
           e.joining_date AS joining_date,
           d.name AS department,
           m.employee_id AS manager_id,
           m.name AS manager_name,
           collect(DISTINCT s.name) AS skills
    ORDER BY e.name
    """
    records = [dict(r) for r in session.run(cql_name, name=clean_term)]

    if not records:
        return {
            "status": "not_found",
            "employee": None,
            "candidates": [],
            "message": f"No employee found matching '{clean_term}'.",
        }

    if len(records) == 1:
        return {
            "status": "exact",
            "employee": records[0],
            "candidates": [],
            "message": f"Found employee record for {records[0]['name']} ({records[0]['employee_id']}).",
        }

    # Multiple matches - provide choices for clarification
    return {
        "status": "multiple",
        "employee": None,
        "candidates": records,
        "message": f"Multiple employees matched '{clean_term}'. Please specify by employee ID or full name.",
    }


def get_employee_manager(session, identifier: str) -> Dict[str, Any]:
    """Retrieve an employee's direct reporting manager."""
    lookup = get_employee_by_id_or_name(session, identifier)
    if lookup["status"] in ("not_found", "multiple"):
        return lookup

    emp = lookup["employee"]
    mgr_id = emp.get("manager_id")
    mgr_name = emp.get("manager_name")

    if not mgr_id:
        return {
            "status": "no_manager",
            "employee": emp,
            "manager": None,
            "message": f"{emp['name']} ({emp['employee_id']}) is a top-level executive with no reporting manager.",
        }

    mgr_lookup = get_employee_by_id_or_name(session, mgr_id)
    return {
        "status": "success",
        "employee": emp,
        "manager": mgr_lookup.get("employee"),
        "message": f"{emp['name']} reports to {mgr_name} ({mgr_id}).",
    }


def get_direct_reports(session, identifier: str) -> Dict[str, Any]:
    """Retrieve all direct reports for a manager."""
    lookup = get_employee_by_id_or_name(session, identifier)
    if lookup["status"] in ("not_found", "multiple"):
        return lookup

    mgr = lookup["employee"]
    cql = """
    MATCH (sub:Employee)-[:REPORTS_TO]->(m:Employee {employee_id: $mgr_id})
    OPTIONAL MATCH (sub)-[:WORKS_IN]->(d:Department)
    OPTIONAL MATCH (sub)-[:HAS_SKILL]->(s:Skill)
    RETURN sub.employee_id AS employee_id,
           sub.name AS name,
           sub.designation AS designation,
           sub.salary AS salary,
           sub.location AS location,
           d.name AS department,
           collect(DISTINCT s.name) AS skills
    ORDER BY sub.name
    """
    reports = [dict(r) for r in session.run(cql, mgr_id=mgr["employee_id"])]

    return {
        "status": "success",
        "manager": mgr,
        "reports": reports,
        "count": len(reports),
        "message": f"Found {len(reports)} direct report(s) for {mgr['name']} ({mgr['employee_id']})."
        if reports
        else f"{mgr['name']} ({mgr['employee_id']}) has no direct reports.",
    }


def get_department_headcounts(session) -> List[Dict[str, Any]]:
    """Retrieve headcount aggregated by department."""
    cql = """
    MATCH (d:Department)
    OPTIONAL MATCH (e:Employee)-[:WORKS_IN]->(d)
    RETURN d.name AS department, count(e) AS count
    ORDER BY count DESC, d.name
    """
    return [dict(r) for r in session.run(cql)]


def build_cypher_query(
    request: EmployeeQueryRequest, is_count: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """Build safe, fully-parameterized Cypher query with strict allowlists.

    Supports dynamic field projection, relationship joining, and database-level LIMIT.

    Args:
        request: Validated EmployeeQueryRequest.
        is_count: If True, produces an optimized count query without projections, joins,
                  or limits.

    Returns:
        tuple: (cypher_string, query_params)
    """
    params: Dict[str, Any] = {}
    where_parts: List[str] = []
    param_idx = 0
    skills_filter = None

    for f in request.filters:
        if f.field not in ALLOWED_FILTER_FIELDS:
            continue

        if f.field == "skills":
            # Handled via UNWIND / count(DISTINCT sk)
            if isinstance(f.value, list):
                skills_filter = [str(s).strip().lower() for s in f.value if str(s).strip()]
            elif isinstance(f.value, str):
                skills_filter = [s.strip().lower() for s in f.value.split(",") if s.strip()]
            continue

        prop_expr = CYPHER_PROPERTY_MAP.get(f.field)
        if not prop_expr:
            continue

        p_name = f"p{param_idx}"
        param_idx += 1

        clause = None
        if f.operator == FilterOperator.EQUALS:
            if isinstance(f.value, str):
                clause = f"toLower({prop_expr}) = toLower(${p_name})"
                params[p_name] = f.value
            else:
                clause = f"{prop_expr} = ${p_name}"
                params[p_name] = f.value

        elif f.operator == FilterOperator.NOT_EQUALS:
            if isinstance(f.value, str):
                clause = f"toLower({prop_expr}) <> toLower(${p_name})"
                params[p_name] = f.value
            else:
                clause = f"{prop_expr} <> ${p_name}"
                params[p_name] = f.value

        elif f.operator == FilterOperator.GREATER_THAN:
            clause = f"{prop_expr} > ${p_name}"
            params[p_name] = f.value

        elif f.operator == FilterOperator.GREATER_THAN_OR_EQUAL:
            clause = f"{prop_expr} >= ${p_name}"
            params[p_name] = f.value

        elif f.operator == FilterOperator.LESS_THAN:
            clause = f"{prop_expr} < ${p_name}"
            params[p_name] = f.value

        elif f.operator == FilterOperator.LESS_THAN_OR_EQUAL:
            clause = f"{prop_expr} <= ${p_name}"
            params[p_name] = f.value

        elif f.operator == FilterOperator.BETWEEN:
            if isinstance(f.value, (list, tuple)) and len(f.value) == 2:
                p_high = f"p{param_idx}"
                param_idx += 1
                clause = f"({prop_expr} >= ${p_name} AND {prop_expr} <= ${p_high})"
                params[p_name] = f.value[0]
                params[p_high] = f.value[1]

        elif f.operator == FilterOperator.IN:
            if isinstance(f.value, list):
                if f.field in ("location", "department", "designation", "name"):
                    clause = f"toLower({prop_expr}) IN ${p_name}"
                    params[p_name] = [str(v).lower() for v in f.value]
                else:
                    clause = f"{prop_expr} IN ${p_name}"
                    params[p_name] = f.value

        elif f.operator == FilterOperator.CONTAINS:
            clause = f"toLower({prop_expr}) CONTAINS toLower(${p_name})"
            params[p_name] = str(f.value)

        if clause:
            if f.negate:
                clause = f"NOT ({clause})"
            where_parts.append(clause)

    # Combine WHERE conditions
    combine_op = " OR " if request.combine_with.upper() == "OR" else " AND "
    where_clause = ("WHERE " + combine_op.join(where_parts)) if where_parts else ""

    # Skills filter WITH clause
    skills_with_clause = ""
    if skills_filter:
        params["req_skills"] = skills_filter
        params["req_skills_count"] = len(skills_filter)
        skills_with_clause = """
        WITH e, d
        UNWIND $req_skills AS req_skill
        MATCH (e)-[:HAS_SKILL]->(sk:Skill)
        WHERE toLower(sk.name) = req_skill
        WITH e, d, count(DISTINCT sk) AS matched_skills
        WHERE matched_skills = $req_skills_count
        """

    # For COUNT operations or when is_count is True
    if is_count or request.operation == OperationType.COUNT:
        cql = f"""
        MATCH (e:Employee)
        OPTIONAL MATCH (e)-[:WORKS_IN]->(d:Department)
        {where_clause}
        {skills_with_clause}
        RETURN count(DISTINCT e) AS count
        """
        return cql.strip(), params

    # Resolve projection fields safely using allowlist
    raw_fields = request.fields if request.fields else DEFAULT_LIST_FIELDS
    selected_fields = [f for f in raw_fields if f in FIELD_EXPRESSION_MAP]
    if not selected_fields:
        selected_fields = list(DEFAULT_LIST_FIELDS)

    # Check which relationships are needed
    filter_fields = {f.field for f in request.filters}
    sort_field = request.sort.field if request.sort else None

    needs_dept = (
        "department" in selected_fields
        or "department" in filter_fields
        or sort_field == "department"
    )
    needs_manager = (
        "manager_id" in selected_fields
        or "manager_name" in selected_fields
        or "manager_id" in filter_fields
        or "manager_name" in filter_fields
        or sort_field in ("manager_id", "manager_name")
    )
    needs_skills = "skills" in selected_fields

    # Build optional matches
    dept_match = "OPTIONAL MATCH (e)-[:WORKS_IN]->(d:Department)" if needs_dept else ""
    mgr_match = "OPTIONAL MATCH (e)-[:REPORTS_TO]->(m:Employee)" if needs_manager else ""
    skills_match = "OPTIONAL MATCH (e)-[:HAS_SKILL]->(s:Skill)" if needs_skills else ""

    # Prepare WITH clause if skills collection is required
    with_clause = ""
    if needs_skills:
        with_items = ["e"]
        if needs_dept:
            with_items.append("d")
        if needs_manager:
            with_items.append("m")
        with_items.append("collect(DISTINCT s.name) AS skills")
        with_clause = f"WITH {', '.join(with_items)}"

    # Construct RETURN projections
    projections = [FIELD_EXPRESSION_MAP[f] for f in selected_fields]
    return_clause = f"RETURN {', '.join(projections)}"

    # Safe Sorting
    order_clause = "ORDER BY e.salary DESC, e.name ASC"
    if request.sort and request.sort.field in ALLOWED_SORT_FIELDS:
        sort_prop = CYPHER_PROPERTY_MAP.get(request.sort.field, f"e.{request.sort.field}")
        sort_dir = "DESC" if request.sort.direction.lower() == "desc" else "ASC"
        order_clause = f"ORDER BY {sort_prop} {sort_dir}"

    params["limit"] = request.limit

    query_parts = [
        "MATCH (e:Employee)",
        dept_match,
        where_clause,
        skills_with_clause,
        mgr_match,
        skills_match,
        with_clause,
        return_clause,
        order_clause,
        "LIMIT $limit",
    ]
    # Filter out empty strings
    cql = "\n    ".join(p for p in query_parts if p.strip())
    return cql.strip(), params


def _build_filter_summary(filters: List[FilterCondition]) -> str:
    """Generate concise human-readable description of active filters."""
    parts = []
    for f in filters:
        if f.field == "location":
            if f.operator == FilterOperator.NOT_EQUALS or f.negate:
                parts.append(f"outside {f.value}")
            elif f.operator == FilterOperator.IN and isinstance(f.value, list):
                parts.append(f"in ({' or '.join(f.value)})")
            else:
                parts.append(f"in {f.value}")
        elif f.field == "department":
            parts.append(f"in {f.value} Department")
        elif f.field == "skills":
            skills_str = ", ".join(f.value) if isinstance(f.value, list) else str(f.value)
            parts.append(f"skilled in ({skills_str})")
        elif f.field == "salary":
            if f.operator == FilterOperator.BETWEEN and isinstance(f.value, list):
                parts.append(f"earning between {_format_inr(f.value[0])} and {_format_inr(f.value[1])}")
            elif f.operator == FilterOperator.GREATER_THAN:
                parts.append(f"earning > {_format_inr(f.value)}")
            elif f.operator == FilterOperator.GREATER_THAN_OR_EQUAL:
                parts.append(f"earning >= {_format_inr(f.value)}")
            elif f.operator == FilterOperator.LESS_THAN:
                parts.append(f"earning < {_format_inr(f.value)}")
            elif f.operator == FilterOperator.LESS_THAN_OR_EQUAL:
                parts.append(f"earning <= {_format_inr(f.value)}")
            else:
                parts.append(f"earning {_format_inr(f.value)}")
        elif f.field == "age":
            if f.operator == FilterOperator.LESS_THAN:
                parts.append(f"aged under {f.value}")
            elif f.operator == FilterOperator.GREATER_THAN:
                parts.append(f"aged above {f.value}")
            else:
                parts.append(f"aged {f.value}")
    return " ".join(parts) if parts else "specified criteria"


def execute_employee_query(request: EmployeeQueryRequest, session) -> Dict[str, Any]:
    """Execute validated EmployeeQueryRequest against Neo4j session.

    Returns:
        dict: Standardized result envelope with status, query_type, data, count, message.
    """
    if request.needs_clarification:
        return {
            "status": "clarification_needed",
            "query_type": "clarification",
            "data": [],
            "count": 0,
            "is_truncated": False,
            "message": request.clarification_message or "Please clarify your employee query.",
        }

    # 1. PROFILE OPERATION
    if request.operation == OperationType.PROFILE:
        target = request.target_identifier or ""
        lookup = get_employee_by_id_or_name(session, target)
        return {
            "status": lookup["status"],
            "query_type": "employee_profile",
            "data": lookup,
            "count": 1 if lookup.get("employee") else len(lookup.get("candidates", [])),
            "is_truncated": False,
            "message": lookup["message"],
        }

    # 2. GET_FIELD OPERATION
    if request.operation == OperationType.GET_FIELD:
        target = request.target_identifier or ""
        lookup = get_employee_by_id_or_name(session, target)
        if lookup["status"] in ("not_found", "multiple"):
            return {
                "status": lookup["status"],
                "query_type": "employee_profile",
                "data": lookup,
                "count": len(lookup.get("candidates", [])),
                "is_truncated": False,
                "message": lookup["message"],
            }
        emp = lookup["employee"]
        fields = request.fields
        field_strs = []
        for f in fields:
            val = emp.get(f)
            if f == "salary":
                val = _format_inr(val)
            elif f == "skills" and isinstance(val, list):
                val = ", ".join(val)
            field_strs.append(f"{f.replace('_', ' ').title()}: {val}")
        return {
            "status": "success",
            "query_type": "get_field",
            "data": {f: emp.get(f) for f in fields},
            "employee": emp,
            "count": 1,
            "is_truncated": False,
            "message": f"{emp['name']} ({emp['employee_id']}) - " + " | ".join(field_strs),
        }

    # 3. MANAGER LOOKUP
    if request.operation == OperationType.MANAGER_LOOKUP:
        target = request.target_identifier or ""
        res = get_employee_manager(session, target)
        return {
            "status": "success" if res["status"] in ("success", "no_manager") else res["status"],
            "query_type": "manager_lookup",
            "data": res,
            "count": 1 if res.get("manager") else 0,
            "is_truncated": False,
            "message": res["message"],
        }

    # 4. DIRECT REPORTS
    if request.operation == OperationType.DIRECT_REPORTS:
        target = request.target_identifier or ""
        res = get_direct_reports(session, target)
        return {
            "status": res["status"],
            "query_type": "direct_reports",
            "data": res,
            "count": res.get("count", 0),
            "is_truncated": False,
            "message": res["message"],
        }

    # 5. DEPARTMENT AGGREGATIONS
    if request.operation == OperationType.AGGREGATE_DEPARTMENT:
        breakdown = get_department_headcounts(session)
        if request.sort and request.sort.field == "count" and request.limit == 1:
            top_dept = breakdown[0] if breakdown else {"department": "None", "count": 0}
            return {
                "status": "success",
                "query_type": "department_highest",
                "data": [top_dept],
                "count": top_dept["count"],
                "is_truncated": False,
                "message": f"The department with the highest headcount is {top_dept['department']} with {top_dept['count']} employee(s).",
            }
        total = sum(d["count"] for d in breakdown)
        return {
            "status": "success",
            "query_type": "headcount_all",
            "data": breakdown,
            "count": total,
            "is_truncated": False,
            "message": f"Total headcount across all departments: {total} employees.",
        }

    # 6. COUNT OPERATION (Fixes Pune headcount bug and all filter counting)
    if request.operation == OperationType.COUNT:
        cql, params = build_cypher_query(request)
        rec = session.run(cql, **params).single()
        count = rec["count"] if rec else 0
        crit_desc = _build_filter_summary(request.filters)
        is_dept_count = any(f.field == "department" for f in request.filters)
        q_type = "headcount" if is_dept_count else "count"
        return {
            "status": "success",
            "query_type": q_type,
            "data": [{"count": count}],
            "count": count,
            "is_truncated": False,
            "message": f"Found {count} employee(s) {crit_desc}." if crit_desc != "specified criteria" else f"Total employee count: {count}.",
        }

    # 7. LIST AND COUNT_AND_LIST
    # 1. Execute separate lightweight count query for true total count
    count_cql, count_params = build_cypher_query(request, is_count=True)
    count_rec = session.run(count_cql, **count_params).single()
    total_count = count_rec["count"] if count_rec else 0

    crit_desc = _build_filter_summary(request.filters)
    query_type = "count_and_list" if request.operation == OperationType.COUNT_AND_LIST else "filtered_list"

    # Determine requested output fields
    raw_fields = request.fields if request.fields else DEFAULT_LIST_FIELDS
    selected_fields = [f for f in raw_fields if f in FIELD_EXPRESSION_MAP]
    if not selected_fields:
        selected_fields = list(DEFAULT_LIST_FIELDS)

    if total_count == 0:
        return {
            "status": "empty",
            "query_type": query_type,
            "data": [],
            "count": 0,
            "is_truncated": False,
            "message": f"No employees found {crit_desc}.",
            "fields": selected_fields,
        }

    # 2. Execute projection data query with LIMIT at the database level
    data_cql, data_params = build_cypher_query(request, is_count=False)
    returned_records = [dict(r) for r in session.run(data_cql, **data_params)]
    is_truncated = total_count > len(returned_records)

    trunc_msg = f" (Displaying first {len(returned_records)} of {total_count})" if is_truncated else ""
    msg = f"Found {total_count} employee(s) {crit_desc}.{trunc_msg}"

    return {
        "status": "success",
        "query_type": query_type,
        "data": returned_records,
        "count": total_count,
        "is_truncated": is_truncated,
        "message": msg,
        "fields": selected_fields,
    }


def handle_employee_query(query: str, session, client: Optional[Any] = None) -> Dict[str, Any]:
    """Top-level handler: Interprets natural language employee query and executes against Neo4j.

    Args:
        query: User's natural language input.
        session: Active Neo4j session.
        client: Optional google.genai Client.

    Returns:
        Standardized result dictionary for Streamlit or programmatic consumption.
    """
    clean_q = query.strip()
    if not clean_q:
        return {
            "status": "empty",
            "query_type": "empty",
            "data": [],
            "count": 0,
            "is_truncated": False,
            "message": "Please enter a question about employees.",
        }

    # Interpret query
    validated_request = interpret_employee_query(clean_q, client=client)

    # Execute validated request
    return execute_employee_query(validated_request, session)


# =====================================================================
# Backward Compatibility Helpers (Preserved for existing imports/tests)
# =====================================================================

def get_filtered_employees(
    session,
    department: Optional[str] = None,
    location: Optional[str] = None,
    designation: Optional[str] = None,
    skills: Optional[List[str]] = None,
    min_salary: Optional[int] = None,
    max_salary: Optional[int] = None,
    limit: int = MAX_RESULTS_LIMIT,
) -> Tuple[List[Dict[str, Any]], bool, int]:
    """Perform multi-criteria filtered search on employees (backward-compatibility layer)."""
    filters = []
    if department:
        filters.append(FilterCondition(field="department", operator=FilterOperator.EQUALS, value=department))
    if location:
        filters.append(FilterCondition(field="location", operator=FilterOperator.EQUALS, value=location))
    if designation:
        filters.append(FilterCondition(field="designation", operator=FilterOperator.CONTAINS, value=designation))
    if skills:
        filters.append(FilterCondition(field="skills", operator=FilterOperator.IN, value=skills))
    if min_salary is not None and max_salary is not None:
        filters.append(FilterCondition(field="salary", operator=FilterOperator.BETWEEN, value=[min_salary, max_salary]))
    elif min_salary is not None:
        filters.append(FilterCondition(field="salary", operator=FilterOperator.GREATER_THAN_OR_EQUAL, value=min_salary))
    elif max_salary is not None:
        filters.append(FilterCondition(field="salary", operator=FilterOperator.LESS_THAN_OR_EQUAL, value=max_salary))

    req = EmployeeQueryRequest(
        operation=OperationType.LIST,
        filters=filters,
        limit=limit,
    )
    
    res = execute_employee_query(req, session)
    return res.get("data", []), res.get("is_truncated", False), res.get("count", 0)


def parse_salary_threshold(query: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract minimum and/or maximum salary in INR from user query (backward-compatibility layer)."""
    min_sal = None
    max_sal = None
    q = query.lower()

    above_lakh = re.search(r"(?:above|greater than|more than|>|at least)\s+(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lpa|l)", q)
    if above_lakh:
        min_sal = int(float(above_lakh.group(1)) * 100000)

    below_lakh = re.search(r"(?:below|less than|under|<|at most)\s+(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lpa|l)", q)
    if below_lakh:
        max_sal = int(float(below_lakh.group(1)) * 100000)

    if not min_sal:
        above_num = re.search(r"(?:above|greater than|more than|>|at least)\s+(\d{6,})", q)
        if above_num:
            min_sal = int(above_num.group(1))

    if not max_sal:
        below_num = re.search(r"(?:below|less than|under|<|at most)\s+(\d{6,})", q)
        if below_num:
            max_sal = int(below_num.group(1))

    return min_sal, max_sal


def parse_skills_from_query(query: str) -> List[str]:
    """Extract candidate skills from query (backward-compatibility layer)."""
    q = query.lower()
    known_skills = [
        "python", "sql", "react", "docker", "kubernetes", "aws", "fastapi",
        "django", "tableau", "pytorch", "mlops", "pandas", "scikit-learn",
        "typescript", "javascript", "terraform", "power bi", "agile", "git",
        "figma", "deep learning", "machine learning", "seo",
    ]
    detected = []
    for skill in known_skills:
        if re.search(rf"\b{re.escape(skill)}\b", q):
            detected.append(skill)
    return detected


def parse_department_from_query(query: str) -> Optional[str]:
    """Extract recognized department from query (backward-compatibility layer)."""
    q = query.lower()
    mapping = {
        "engineering": "Engineering",
        "data science": "Data Science",
        "human resources": "Human Resources",
        "hr": "Human Resources",
        "finance": "Finance",
        "marketing": "Marketing",
        "product": "Product",
    }
    for key, dept_name in mapping.items():
        if re.search(rf"\b{re.escape(key)}\b", q):
            return dept_name
    return None


def parse_location_from_query(query: str) -> Optional[str]:
    """Extract recognized location from query (backward-compatibility layer)."""
    q = query.lower()
    mapping = {
        "hyderabad": "Hyderabad",
        "bengaluru": "Bengaluru",
        "bangalore": "Bengaluru",
        "pune": "Pune",
        "mumbai": "Mumbai",
        "gurugram": "Gurugram",
        "gurgaon": "Gurugram",
    }
    for key, loc_name in mapping.items():
        if re.search(rf"\b{re.escape(key)}\b", q):
            return loc_name
    return None
