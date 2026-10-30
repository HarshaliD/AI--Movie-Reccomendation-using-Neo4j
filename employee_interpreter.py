"""Employee Natural Language Query Interpreter.

Uses Gemini LLM to interpret natural language employee questions into a strictly validated,
structured EmployeeQueryRequest using Pydantic. Includes a deterministic fallback for offline/malformed
responses and robust Indian salary parsing.
"""

from enum import Enum
import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3.1-flash-lite"

# Graph Schema Allowlists
ALLOWED_FIELDS = {
    "employee_id",
    "name",
    "age",
    "department",
    "designation",
    "salary",
    "location",
    "joining_date",
    "manager_id",
    "manager_name",
    "skills",
    "count",
}

ALLOWED_FILTER_FIELDS = {
    "employee_id",
    "name",
    "age",
    "department",
    "designation",
    "salary",
    "location",
    "joining_date",
    "skills",
}

ALLOWED_SORT_FIELDS = {
    "name",
    "salary",
    "age",
    "joining_date",
    "employee_id",
    "department",
    "location",
    "count",
}


class EntityType(str, Enum):
    EMPLOYEE = "employee"
    DEPARTMENT = "department"
    SKILL = "skill"


class OperationType(str, Enum):
    COUNT = "count"
    LIST = "list"
    PROFILE = "profile"
    GET_FIELD = "get_field"
    MANAGER_LOOKUP = "manager_lookup"
    DIRECT_REPORTS = "direct_reports"
    AGGREGATE_DEPARTMENT = "aggregate_department"
    COUNT_AND_LIST = "count_and_list"


class FilterOperator(str, Enum):
    EQUALS = "="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="
    BETWEEN = "between"
    IN = "in"
    CONTAINS = "contains"


def parse_indian_salary(val: Union[str, int, float]) -> Optional[int]:
    """Normalize Indian salary expressions (e.g. 12 lakh, 12 LPA, 1.5 crore, ₹1,200,000) to integer INR."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return int(round(val))

    s = str(val).strip().lower()
    s = s.replace("₹", "").replace(",", "").strip()

    # Crore check (e.g. 1.5 crore, 1.5 cr, 2 crore)
    cr_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:crore|crores|cr)\b", s)
    if cr_match:
        return int(float(cr_match.group(1)) * 10_000_000)

    # Lakh / LPA check (e.g. 12 lakh, 12 lakhs, 12 lpa, 12 l, 12.5lpa)
    lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lpa|l)\b", s)
    if lakh_match:
        return int(float(lakh_match.group(1)) * 100_000)

    # Plain integer or decimal string
    num_match = re.search(r"\b(\d+(?:\.\d+)?)\b", s)
    if num_match:
        raw_num = float(num_match.group(1))
        # If someone passed single/double digit as salary like 12 when meaning 12 LPA
        if raw_num < 100 and ("lpa" in s or "lakh" in s or "l" in s):
            return int(raw_num * 100_000)
        return int(raw_num)

    return None


class FilterCondition(BaseModel):
    field: str
    operator: FilterOperator = FilterOperator.EQUALS
    value: Union[str, int, float, List[Union[str, int, float]]]
    negate: bool = False

    @field_validator("field")
    @classmethod
    def validate_field(cls, v: str) -> str:
        clean = v.strip().lower()
        # Aliases
        alias_map = {
            "dept": "department",
            "loc": "location",
            "city": "location",
            "role": "designation",
            "title": "designation",
            "pay": "salary",
            "compensation": "salary",
            "skill": "skills",
            "technologies": "skills",
            "tech_stack": "skills",
            "id": "employee_id",
            "emp_id": "employee_id",
        }
        clean = alias_map.get(clean, clean)
        if clean not in ALLOWED_FILTER_FIELDS:
            raise ValueError(f"Field '{clean}' is not supported for filtering. Allowed: {sorted(ALLOWED_FILTER_FIELDS)}")
        return clean

    @model_validator(mode="after")
    def validate_and_normalize_value(self) -> "FilterCondition":
        if self.field == "salary":
            if self.operator == FilterOperator.BETWEEN:
                if isinstance(self.value, (list, tuple)) and len(self.value) == 2:
                    v1 = parse_indian_salary(self.value[0])
                    v2 = parse_indian_salary(self.value[1])
                    if v1 is None or v2 is None:
                        raise ValueError(f"Invalid salary values for between: {self.value}")
                    self.value = [min(v1, v2), max(v1, v2)]
                else:
                    raise ValueError(f"Operator 'between' requires a list of 2 values, got: {self.value}")
            elif isinstance(self.value, (list, tuple)):
                self.value = [parse_indian_salary(v) for v in self.value if parse_indian_salary(v) is not None]
            else:
                sal = parse_indian_salary(self.value)
                if sal is None:
                    raise ValueError(f"Invalid salary value: {self.value}")
                self.value = sal
        elif self.field == "age":
            try:
                if self.operator == FilterOperator.BETWEEN:
                    if isinstance(self.value, (list, tuple)) and len(self.value) == 2:
                        self.value = [int(self.value[0]), int(self.value[1])]
                    else:
                        raise ValueError(f"Operator 'between' on age requires 2 values, got {self.value}")
                elif not isinstance(self.value, (list, tuple)):
                    self.value = int(self.value)
            except (ValueError, TypeError) as err:
                raise ValueError(f"Invalid age value '{self.value}': {err}")
        elif self.field == "skills":
            if isinstance(self.value, str):
                self.value = [s.strip() for s in self.value.split(",") if s.strip()]
            elif isinstance(self.value, list):
                self.value = [str(s).strip() for s in self.value if str(s).strip()]
        return self


class SortSpec(BaseModel):
    field: str = "name"
    direction: Literal["asc", "desc"] = "asc"

    @field_validator("field")
    @classmethod
    def validate_sort_field(cls, v: str) -> str:
        clean = v.strip().lower()
        alias_map = {
            "dept": "department",
            "loc": "location",
            "pay": "salary",
            "compensation": "salary",
            "id": "employee_id",
        }
        clean = alias_map.get(clean, clean)
        if clean not in ALLOWED_SORT_FIELDS:
            raise ValueError(f"Sort field '{clean}' is not supported. Allowed: {sorted(ALLOWED_SORT_FIELDS)}")
        return clean


class EmployeeQueryRequest(BaseModel):
    entity: EntityType = EntityType.EMPLOYEE
    operation: OperationType = OperationType.LIST
    target_identifier: Optional[str] = None
    fields: List[str] = Field(
        default_factory=lambda: [
            "employee_id",
            "name",
            "designation",
            "department",
            "location",
            "salary",
            "skills",
        ]
    )
    filters: List[FilterCondition] = Field(default_factory=list)
    combine_with: Literal["AND", "OR"] = "AND"
    sort: Optional[SortSpec] = None
    limit: int = Field(default=20, ge=1, le=100)
    include_count: bool = False
    needs_clarification: bool = False
    clarification_message: Optional[str] = None
    raw_query: Optional[str] = None

    @field_validator("fields")
    @classmethod
    def validate_fields(cls, v: List[str]) -> List[str]:
        if not v:
            return [
                "employee_id",
                "name",
                "designation",
                "department",
                "location",
                "salary",
                "skills",
            ]
        valid_fields = []
        for f in v:
            clean = f.strip().lower()
            if clean in ALLOWED_FIELDS:
                valid_fields.append(clean)
        return valid_fields or ["employee_id", "name"]


INTERPRETER_SYSTEM_PROMPT = """You are a natural language query interpreter for an Employee Knowledge Graph in Neo4j.
Your task is to convert the user's natural language question into a validated JSON request matching the strict schema below.

Graph Schema:
- Nodes:
  - (:Employee {employee_id: String, name: String, age: Integer, designation: String, salary: Integer (in INR), location: String, joining_date: String})
  - (:Department {name: String})
  - (:Skill {name: String})
- Relationships:
  - (e:Employee)-[:WORKS_IN]->(d:Department)
  - (e:Employee)-[:HAS_SKILL]->(s:Skill)
  - (e:Employee)-[:REPORTS_TO]->(m:Employee)  (Note: e reports to manager m)

Allowed Operations:
- "count": Count matching records (e.g., "How many employees are in Pune?")
- "list": List matching employee records (e.g., "List Python developers in Pune")
- "profile": View details of a specific employee by ID or name (e.g., "Tell me about EMP001", "Who is Priya?")
- "get_field": Retrieve specific fields of a specific employee (e.g., "What skills does EMP001 have?", "What is Rahul's salary?")
- "manager_lookup": Find who manages the target employee (e.g., "Who manages Priya?", "Who is EMP001's manager?")
- "direct_reports": Find who reports to the target manager (e.g., "Who reports to Rahul?", "Direct reports of EMP002")
- "aggregate_department": Aggregate employees by department (e.g., "How many employees work in each department?", "Which department has the highest headcount?")
- "count_and_list": When the user asks for BOTH the count and matching names/records (e.g., "How many Python developers are in Pune, and what are their names?")

Supported Filter Fields:
- "location" (String: e.g., "Pune", "Bengaluru", "Hyderabad", "Mumbai", "Gurugram")
- "department" (String: e.g., "Engineering", "Data Science", "Product", "Human Resources", "Finance", "Marketing")
- "skills" (List of Strings: e.g., ["Python", "SQL"])
- "salary" (Integer INR: e.g., 1200000 for 12 lakh / 12 LPA)
- "age" (Integer: e.g., 30)
- "designation" (String: e.g., "Software Engineer")
- "employee_id" (String: e.g., "EMP001")
- "name" (String: e.g., "Priya")

Operators:
- "=": exact match
- "!=": not equal (e.g. "employees who are not in Pune")
- ">": strictly greater than (e.g. "earning more than 12 lakh", "above 12 lakh")
- ">=": greater than or equal to (e.g. "earning at least 12 lakh")
- "<": strictly less than (e.g. "aged below 30", "under 15 lakh", "less than 20 lakh")
- "<=": less than or equal to (e.g. "at most 20 lakh")
- "between": between two values inclusive [min, max] (e.g. "earning between 12 lakh and 20 lakh")
- "in": value is in a list (e.g. "in Pune or Hyderabad" -> ["Pune", "Hyderabad"])
- "contains": string substring match

Indian Salary Normalization:
- 12 lakh / 12 LPA / 12L = 1200000
- 1.5 crore = 15000000

Rules:
1. STRICT OPERATORS: "more than" / "above" / "greater than" MUST use ">". "at least" MUST use ">=". "less than" / "below" / "under" MUST use "<". "at most" MUST use "<=".
2. DO NOT hallucinate properties or Cypher code.
3. Return ONLY a valid JSON object matching this structure:
{
  "entity": "employee",
  "operation": "count" | "list" | "profile" | "get_field" | "manager_lookup" | "direct_reports" | "aggregate_department" | "count_and_list",
  "target_identifier": "EMP001" or "Priya" (or null if not applicable),
  "fields": ["name", "salary", "skills"],
  "filters": [
    {
      "field": "location",
      "operator": "=",
      "value": "Pune",
      "negate": false
    }
  ],
  "combine_with": "AND" | "OR",
  "sort": {"field": "salary", "direction": "desc"} (or null),
  "limit": 20,
  "include_count": false,
  "needs_clarification": false,
  "clarification_message": null
}
"""


def extract_json_from_llm_response(text: str) -> Dict[str, Any]:
    """Safely extract JSON from raw LLM output."""
    clean = text.strip()
    # Check for markdown code fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean, re.IGNORECASE)
    if fence_match:
        clean = fence_match.group(1).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Try finding outer braces
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = clean[start : end + 1]
            return json.loads(snippet)
        raise


def deterministic_fallback_parser(query: str) -> EmployeeQueryRequest:
    """Deterministic rule-based parser for offline execution, simple queries, or LLM fallback."""
    q = query.strip()
    norm = q.lower()

    # 1. Manager lookup
    mgr_pattern = re.search(
        r"(?:who is\s+)?(?:the\s+)?manager\s+(?:of|for)\s+([a-zA-Z0-9\s]+)|([a-zA-Z0-9]+)(?:'s)?\s+manager|who\s+does\s+([a-zA-Z0-9\s]+)\s+report\s+to|who\s+manages\s+([a-zA-Z0-9\s]+)",
        q,
        re.IGNORECASE,
    )
    if mgr_pattern:
        target = (
            mgr_pattern.group(1)
            or mgr_pattern.group(2)
            or mgr_pattern.group(3)
            or mgr_pattern.group(4)
            or ""
        ).strip()
        target = re.sub(r"[?'\.]", "", target).strip()
        if target:
            return EmployeeQueryRequest(
                operation=OperationType.MANAGER_LOOKUP,
                target_identifier=target,
                raw_query=q,
            )

    # 2. Direct reports
    rep_pattern = re.search(
        r"(?:who\s+reports\s+(?:directly\s+)?to|direct\s+reports\s+(?:of|for)|subordinates\s+of)\s+([a-zA-Z0-9\s]+)",
        q,
        re.IGNORECASE,
    )
    if rep_pattern:
        target = rep_pattern.group(1).strip()
        target = re.sub(r"[?'\.]", "", target).strip()
        if target:
            return EmployeeQueryRequest(
                operation=OperationType.DIRECT_REPORTS,
                target_identifier=target,
                raw_query=q,
            )

    # 3. Department Headcount Aggregation
    if "each department" in norm or "by department" in norm or "per department" in norm:
        return EmployeeQueryRequest(
            operation=OperationType.AGGREGATE_DEPARTMENT,
            raw_query=q,
        )
    if "highest headcount" in norm or "most employees" in norm:
        return EmployeeQueryRequest(
            operation=OperationType.AGGREGATE_DEPARTMENT,
            sort=SortSpec(field="count", direction="desc"),
            limit=1,
            raw_query=q,
        )

    # 4. Check for Specific Employee ID (e.g. EMP001)
    emp_id_match = re.search(r"\b(EMP\d{3})\b", q, re.IGNORECASE)
    if emp_id_match:
        emp_id = emp_id_match.group(1).upper()
        if "skill" in norm:
            return EmployeeQueryRequest(
                operation=OperationType.GET_FIELD,
                target_identifier=emp_id,
                fields=["skills"],
                raw_query=q,
            )
        if "salary" in norm or "earn" in norm:
            return EmployeeQueryRequest(
                operation=OperationType.GET_FIELD,
                target_identifier=emp_id,
                fields=["salary"],
                raw_query=q,
            )
        if any(w in norm for w in ["who is", "tell me about", "profile of", "about", "show", "details of"]):
            return EmployeeQueryRequest(
                operation=OperationType.PROFILE,
                target_identifier=emp_id,
                raw_query=q,
            )

    # 5. Extract filters (location, dept, skills, salary, age)
    filters: List[FilterCondition] = []
    combine_with = "AND"

    # Known locations
    known_locations = {
        "pune": "Pune",
        "hyderabad": "Hyderabad",
        "bengaluru": "Bengaluru",
        "bangalore": "Bengaluru",
        "mumbai": "Mumbai",
        "gurugram": "Gurugram",
        "gurgaon": "Gurugram",
    }
    found_locs = []
    for loc_key, loc_name in known_locations.items():
        if re.search(rf"\b{re.escape(loc_key)}\b", norm):
            if loc_name not in found_locs:
                found_locs.append(loc_name)

    if found_locs:
        # Check for negation ("not in Pune", "outside Pune")
        is_negated = bool(re.search(r"\b(?:not in|outside|excluding)\s+(?:" + "|".join(known_locations.keys()) + r")\b", norm))
        if len(found_locs) == 1:
            filters.append(
                FilterCondition(
                    field="location",
                    operator=FilterOperator.NOT_EQUALS if is_negated else FilterOperator.EQUALS,
                    value=found_locs[0],
                )
            )
        else:
            # "Pune or Hyderabad"
            filters.append(
                FilterCondition(
                    field="location",
                    operator=FilterOperator.IN,
                    value=found_locs,
                    negate=is_negated,
                )
            )

    # Known departments
    known_departments = {
        "engineering": "Engineering",
        "data science": "Data Science",
        "human resources": "Human Resources",
        "hr": "Human Resources",
        "finance": "Finance",
        "marketing": "Marketing",
        "product": "Product",
    }
    for dept_key, dept_name in known_departments.items():
        if re.search(rf"\b{re.escape(dept_key)}\b", norm):
            filters.append(
                FilterCondition(
                    field="department",
                    operator=FilterOperator.EQUALS,
                    value=dept_name,
                )
            )
            break

    # Known skills
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
    ]
    found_skills = []
    for skill in known_skills:
        if re.search(rf"\b{re.escape(skill)}\b", norm):
            found_skills.append(skill.title() if len(skill) > 3 else skill.upper())
    if found_skills:
        filters.append(
            FilterCondition(
                field="skills",
                operator=FilterOperator.IN,
                value=found_skills,
            )
        )

    # Age filter (checked first to prevent confusing age thresholds with salary numbers)
    age_below = re.search(r"\b(?:aged?\s+)?(?:below|under|less than|<)\s+(\d{1,2})\b", norm)
    if age_below and ("age" in norm or not any(k in norm for k in ["salary", "lakh", "lpa", "earn", "₹"])):
        filters.append(FilterCondition(field="age", operator=FilterOperator.LESS_THAN, value=int(age_below.group(1))))

    age_above = re.search(r"\b(?:aged?\s+)?(?:above|greater than|>)\s+(\d{1,2})\b", norm)
    if age_above and ("age" in norm or not any(k in norm for k in ["salary", "lakh", "lpa", "earn", "₹"])):
        filters.append(FilterCondition(field="age", operator=FilterOperator.GREATER_THAN, value=int(age_above.group(1))))

    # Salary filters
    has_sal_context = any(
        k in norm
        for k in [
            "salary",
            "salaries",
            "earn",
            "earns",
            "earning",
            "paid",
            "pay",
            "compensation",
            "lakh",
            "lakhs",
            "lpa",
            "crore",
            "crores",
            "cr",
            "₹",
        ]
    )

    # Check for "between X and Y"
    between_match = re.search(
        r"(?:between|from)\s+([₹0-9\.\,\s\w]+?)\s+(?:and|to|but less than)\s+([₹0-9\.\,\s\w]+)",
        norm,
    )
    if between_match and has_sal_context:
        s1 = parse_indian_salary(between_match.group(1))
        s2 = parse_indian_salary(between_match.group(2))
        if s1 and s2:
            filters.append(
                FilterCondition(
                    field="salary",
                    operator=FilterOperator.BETWEEN,
                    value=[min(s1, s2), max(s1, s2)],
                )
            )
    elif has_sal_context:
        sal_unit_pat = r"([₹0-9\.\,]+(?:\s*(?:lakh|lakhs|lpa|l|crore|crores|cr))?|\d{6,})"
        gte_match = re.search(rf"(?:at least|minimum of|>=)\s+{sal_unit_pat}", norm)
        gt_match = re.search(rf"(?:more than|above|greater than|>)\s+{sal_unit_pat}", norm)
        lte_match = re.search(rf"(?:at most|maximum of|<=)\s+{sal_unit_pat}", norm)
        lt_match = re.search(rf"(?:less than|below|under|<)\s+{sal_unit_pat}", norm)

        if gte_match:
            sal = parse_indian_salary(gte_match.group(1))
            if sal:
                filters.append(FilterCondition(field="salary", operator=FilterOperator.GREATER_THAN_OR_EQUAL, value=sal))
        elif gt_match:
            sal = parse_indian_salary(gt_match.group(1))
            if sal:
                filters.append(FilterCondition(field="salary", operator=FilterOperator.GREATER_THAN, value=sal))

        if lte_match:
            sal = parse_indian_salary(lte_match.group(1))
            if sal:
                filters.append(FilterCondition(field="salary", operator=FilterOperator.LESS_THAN_OR_EQUAL, value=sal))
        elif lt_match:
            sal = parse_indian_salary(lt_match.group(1))
            if sal:
                filters.append(FilterCondition(field="salary", operator=FilterOperator.LESS_THAN, value=sal))

    # Sort & Limit (e.g. "five highest-paid employees", "top 5")
    sort_spec = None
    limit = 20
    top_n_match = re.search(r"\b(?:top|first|list the)?\s*(\d+|five|three|ten)?\s*(?:highest[- ]paid|highest salary)\b", norm)
    if top_n_match or "highest-paid" in norm or "highest paid" in norm:
        word_to_num = {"three": 3, "five": 5, "ten": 10}
        n_str = top_n_match.group(1) if top_n_match and top_n_match.group(1) else "5"
        limit = int(word_to_num.get(n_str, n_str if n_str.isdigit() else 5))
        sort_spec = SortSpec(field="salary", direction="desc")

    # Determine operation: COUNT, COUNT_AND_LIST, or LIST
    has_count_kw = bool(re.search(r"\b(how many|count employees|headcount)\b", norm))
    has_list_kw = bool(re.search(r"\b(what are their names|names|list|show)\b", norm))

    if has_count_kw and has_list_kw:
        operation = OperationType.COUNT_AND_LIST
    elif has_count_kw:
        operation = OperationType.COUNT
    else:
        operation = OperationType.LIST

    # Requested fields detection
    req_fields = []
    if "name" in norm:
        req_fields.append("name")
    if "salary" in norm or "earning" in norm:
        req_fields.append("salary")
    if "skill" in norm:
        req_fields.append("skills")
    if "department" in norm:
        req_fields.append("department")
    if "location" in norm:
        req_fields.append("location")

    if not req_fields:
        req_fields = [
            "employee_id",
            "name",
            "designation",
            "department",
            "location",
            "salary",
            "skills",
        ]

    # Check for empty filter query that just asks about a person's name
    if not filters and operation == OperationType.LIST and not sort_spec:
        clean_name = re.sub(
            r"\b(who is|tell me about|show|find|profile of|employee|employees)\b",
            "",
            q,
            flags=re.IGNORECASE,
        ).strip()
        clean_name = re.sub(r"[?'\.]", "", clean_name).strip()
        if clean_name and len(clean_name) >= 3 and not any(k in clean_name.lower() for k in ["pune", "engineering", "salary"]):
            return EmployeeQueryRequest(
                operation=OperationType.PROFILE,
                target_identifier=clean_name,
                raw_query=q,
            )

    return EmployeeQueryRequest(
        operation=operation,
        fields=req_fields,
        filters=filters,
        combine_with=combine_with,
        sort=sort_spec,
        limit=limit,
        include_count=(operation == OperationType.COUNT_AND_LIST),
        raw_query=q,
    )


def interpret_employee_query(query: str, client: Optional[Any] = None) -> EmployeeQueryRequest:
    """Interpret natural language question using Gemini or fallback to deterministic parser.

    Args:
        query: User's raw question string.
        client: Optional google.genai Client instance.

    Returns:
        Validated EmployeeQueryRequest instance.
    """
    clean_q = query.strip()
    if not clean_q:
        return EmployeeQueryRequest(
            operation=OperationType.LIST,
            needs_clarification=True,
            clarification_message="Empty question provided. Please ask an employee-related question.",
            raw_query=query,
        )

    # If Gemini client is provided, attempt LLM interpretation
    if client is not None:
        try:
            prompt = f"{INTERPRETER_SYSTEM_PROMPT}\n\nUSER QUESTION: \"{clean_q}\"\n\nJSON RESPONSE:"
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            if response and response.text:
                raw_json = extract_json_from_llm_response(response.text)
                raw_json["raw_query"] = clean_q
                validated = EmployeeQueryRequest(**raw_json)
                print("\n========== GEMINI INTERPRETATION ==========")
                print(validated)
                print("\n========== REQUEST DICTIONARY =============")
                print(validated.model_dump())
                print("===========================================\n")
                print("GEMINI INTERPRETATION SUCCESSFUL")          #Debug
                return validated
        except Exception as err:
            logger.warning("Gemini LLM interpretation failed, falling back to deterministic parser: %s", err)

    # Deterministic fallback parser
    return deterministic_fallback_parser(clean_q)
