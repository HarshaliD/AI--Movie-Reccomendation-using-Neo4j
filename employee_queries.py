"""Employee Knowledge Graph Query Handler.

Executes deterministic, parameterized Cypher queries against Neo4j AuraDB.
No unrestricted LLM Cypher generation.
Supports:
- Employee lookup by ID or name (with disambiguation for multiple matches)
- Manager lookup (and handling top-level executives with no manager)
- Direct reports of a manager
- Department listings and headcounts
- Multi-skill filtering (e.g., Python AND SQL)
- Location filtering (e.g., Hyderabad)
- Salary threshold filtering (e.g., above 12 Lakh)
- Combined multi-criteria search (e.g., Hyderabad-based Python engineers earning above 12 Lakh)
"""

import re
from typing import Any, Dict, List, Optional, Tuple

MAX_RESULTS_LIMIT = 20


def _format_inr(amount: Optional[int]) -> str:
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
    if lookup["status"] == "not_found":
        return lookup
    if lookup["status"] == "multiple":
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
    if lookup["status"] == "not_found":
        return lookup
    if lookup["status"] == "multiple":
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
    """Perform multi-criteria filtered search on employees.

    Returns:
        tuple: (results: list[dict], is_truncated: bool, total_count: int)
    """
    conditions = []
    params: Dict[str, Any] = {"limit": limit}

    if department:
        conditions.append("toLower(d.name) = toLower($dept)")
        params["dept"] = department

    if location:
        conditions.append("toLower(e.location) = toLower($loc)")
        params["loc"] = location

    if designation:
        conditions.append("toLower(e.designation) CONTAINS toLower($desig)")
        params["desig"] = designation

    if min_salary is not None:
        conditions.append("e.salary >= $min_sal")
        params["min_sal"] = min_salary

    if max_salary is not None:
        conditions.append("e.salary <= $max_sal")
        params["max_sal"] = max_salary

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    skills_with_clause = ""
    if skills:
        params["skills"] = [s.strip().lower() for s in skills if s.strip()]
        params["skills_count"] = len(params["skills"])
        skills_with_clause = """
        WITH e, d
        UNWIND $skills AS req_skill
        MATCH (e)-[:HAS_SKILL]->(sk:Skill)
        WHERE toLower(sk.name) = req_skill
        WITH e, d, count(DISTINCT sk) AS matched_skills
        WHERE matched_skills = $skills_count
        """

    cql = f"""
    MATCH (e:Employee)-[:WORKS_IN]->(d:Department)
    {where_clause}
    {skills_with_clause}
    OPTIONAL MATCH (e)-[:REPORTS_TO]->(m:Employee)
    OPTIONAL MATCH (e)-[:HAS_SKILL]->(s:Skill)
    WITH e, d, m, collect(DISTINCT s.name) AS skills
    RETURN e.employee_id AS employee_id,
           e.name AS name,
           e.age AS age,
           e.designation AS designation,
           e.salary AS salary,
           e.location AS location,
           e.joining_date AS joining_date,
           d.name AS department,
           m.name AS manager_name,
           skills
    ORDER BY e.salary DESC, e.name
    """

    all_records = [dict(r) for r in session.run(cql, **params)]
    total_count = len(all_records)
    is_truncated = total_count > limit
    returned_records = all_records[:limit]

    return returned_records, is_truncated, total_count


def parse_salary_threshold(query: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract minimum and/or maximum salary in INR from user query."""
    min_sal = None
    max_sal = None
    q = query.lower()

    above_lakh = re.search(r"(?:above|greater than|more than|>|at least)\s+(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lpa|l)", q)
    if above_lakh:
        val = float(above_lakh.group(1))
        min_sal = int(val * 100000)

    below_lakh = re.search(r"(?:below|less than|under|<|at most)\s+(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lpa|l)", q)
    if below_lakh:
        val = float(below_lakh.group(1))
        max_sal = int(val * 100000)

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
    """Extract candidate skills from query."""
    q = query.lower()
    known_skills = [
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
        "power bi",
        "agile",
        "git",
        "figma",
        "deep learning",
        "machine learning",
        "seo",
    ]
    detected = []
    for skill in known_skills:
        if re.search(rf"\b{re.escape(skill)}\b", q):
            detected.append(skill)
    return detected


def parse_department_from_query(query: str) -> Optional[str]:
    """Extract recognized department from query."""
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
    """Extract recognized location from query."""
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


def handle_employee_query(query: str, session) -> Dict[str, Any]:
    """Route natural language employee query to the appropriate parameterized Cypher handler."""
    clean_q = query.strip()
    norm_q = clean_q.lower()

    # 1. Manager Lookup
    mgr_pattern = re.search(
        r"(?:who is\s+)?(?:the\s+)?manager\s+(?:of|for)\s+([a-zA-Z0-9\s]+)|([a-zA-Z0-9]+)(?:'s)?\s+manager|who\s+does\s+([a-zA-Z0-9\s]+)\s+report\s+to",
        clean_q,
        re.IGNORECASE,
    )
    if mgr_pattern:
        target = (mgr_pattern.group(1) or mgr_pattern.group(2) or mgr_pattern.group(3) or "").strip()
        target = re.sub(r"[?'\.]", "", target).strip()
        if target:
            res = get_employee_manager(session, target)
            return {
                "status": "success" if res["status"] in ("success", "no_manager") else res["status"],
                "query_type": "manager_lookup",
                "data": res,
                "count": 1 if res.get("manager") else 0,
                "is_truncated": False,
                "message": res["message"],
            }

    # 2. Direct Reports
    reports_pattern = re.search(
        r"(?:who\s+reports\s+to|direct\s+reports\s+(?:of|for)|subordinates\s+of)\s+([a-zA-Z0-9\s]+)",
        clean_q,
        re.IGNORECASE,
    )
    if reports_pattern:
        target = reports_pattern.group(1).strip()
        target = re.sub(r"[?'\.]", "", target).strip()
        if target:
            res = get_direct_reports(session, target)
            return {
                "status": res["status"],
                "query_type": "direct_reports",
                "data": res,
                "count": res.get("count", 0),
                "is_truncated": False,
                "message": res["message"],
            }

    # 3. Headcount / Count Query
    if re.search(r"\b(how many employees|headcount|employee count|count employees)\b", norm_q):
        target_dept = parse_department_from_query(norm_q)
        if target_dept:
            cql = """
            MATCH (e:Employee)-[:WORKS_IN]->(d:Department)
            WHERE toLower(d.name) = toLower($dept)
            RETURN d.name AS department, count(e) AS count
            """
            rec = session.run(cql, dept=target_dept).single()
            count = rec["count"] if rec else 0
            return {
                "status": "success",
                "query_type": "headcount",
                "data": [{"department": target_dept, "count": count}],
                "count": count,
                "is_truncated": False,
                "message": f"{count} employee(s) work in {target_dept}.",
            }
        else:
            breakdown = get_department_headcounts(session)
            total = sum(d["count"] for d in breakdown)
            return {
                "status": "success",
                "query_type": "headcount_all",
                "data": breakdown,
                "count": total,
                "is_truncated": False,
                "message": f"Total headcount across all departments: {total} employees.",
            }

    # 4. Direct Employee ID Pattern
    emp_id_match = re.search(r"\b(EMP\d{3})\b", clean_q, re.IGNORECASE)
    if emp_id_match:
        emp_id = emp_id_match.group(1).upper()
        if len(norm_q.split()) <= 4 or "who is" in norm_q or "about" in norm_q or "find" in norm_q:
            lookup = get_employee_by_id_or_name(session, emp_id)
            return {
                "status": lookup["status"],
                "query_type": "employee_profile",
                "data": lookup,
                "count": 1 if lookup.get("employee") else 0,
                "is_truncated": False,
                "message": lookup["message"],
            }

    # 5. Multi-Criteria Filters
    dept = parse_department_from_query(norm_q)
    loc = parse_location_from_query(norm_q)
    skills = parse_skills_from_query(norm_q)
    min_sal, max_sal = parse_salary_threshold(norm_q)

    if dept or loc or skills or min_sal is not None or max_sal is not None:
        records, is_truncated, total = get_filtered_employees(
            session=session,
            department=dept,
            location=loc,
            skills=skills,
            min_salary=min_sal,
            max_salary=max_sal,
            limit=MAX_RESULTS_LIMIT,
        )

        filter_desc = []
        if loc:
            filter_desc.append(f"in {loc}")
        if dept:
            filter_desc.append(f"in {dept}")
        if skills:
            filter_desc.append(f"with skills ({', '.join(skills)})")
        if min_sal:
            filter_desc.append(f"earning >= {_format_inr(min_sal)}")
        if max_sal:
            filter_desc.append(f"earning <= {_format_inr(max_sal)}")

        crit_str = " ".join(filter_desc) if filter_desc else "specified criteria"
        trunc_msg = f" (Displaying first {len(records)} of {total})" if is_truncated else ""
        msg = f"Found {total} employee(s) {crit_str}.{trunc_msg}" if total > 0 else f"No employees found {crit_str}."

        return {
            "status": "success" if total > 0 else "empty",
            "query_type": "filtered_list",
            "data": records,
            "count": total,
            "is_truncated": is_truncated,
            "message": msg,
        }

    # 6. Fallback Name Search
    name_clean = re.sub(r"\b(who is|tell me about|show|find|profile of|employee)\b", "", norm_q, flags=re.IGNORECASE).strip()
    name_clean = re.sub(r"[?'\.]", "", name_clean).strip()
    if name_clean and len(name_clean) >= 3:
        lookup = get_employee_by_id_or_name(session, name_clean)
        if lookup["status"] != "not_found":
            return {
                "status": lookup["status"],
                "query_type": "employee_profile",
                "data": lookup,
                "count": 1 if lookup.get("employee") else len(lookup.get("candidates", [])),
                "is_truncated": False,
                "message": lookup["message"],
            }

    return {
        "status": "empty",
        "query_type": "unrecognized",
        "data": [],
        "count": 0,
        "is_truncated": False,
        "message": "Could not identify specific employee criteria. You can search by employee ID (e.g., EMP001), name, department, location, skills, or salary threshold.",
    }
