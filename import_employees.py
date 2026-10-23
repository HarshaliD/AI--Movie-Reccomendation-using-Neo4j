"""Excel-to-Neo4j Ingestion for Employee Knowledge Graph.

Imports data/employees.xlsx into Neo4j AuraDB with strict validation:
- Validates columns, types, duplicate IDs, missing fields, manager references
- Detects self-reporting and circular reporting chains
- Merges Employee, Department, and Skill nodes idempotently
- Preserves existing Movie recommendation data intact
"""

import argparse
import os
from pathlib import Path
import sys
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
MOVIE_REC_DIR = ROOT_DIR / "movie-recommendation"
if str(MOVIE_REC_DIR) not in sys.path:
    sys.path.insert(0, str(MOVIE_REC_DIR))

from app import DATABASE, get_driver


def find_excel_file(specified_path=None):
    """Resolve the employee Excel workbook path to the canonical data/employees.xlsx."""
    if specified_path:
        path = Path(specified_path).resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"Specified Excel file not found: {specified_path}")

    canonical_path = ROOT_DIR / "data" / "employees.xlsx"
    if canonical_path.is_file():
        return canonical_path.resolve()

    raise FileNotFoundError(f"Could not find canonical workbook at: {canonical_path}")


def validate_employees_data(df):
    """Validate raw employee records from Excel before performing any DB operations.

    Returns:
        tuple: (is_valid: bool, errors: list[str], cleaned_records: list[dict])
    """
    errors = []
    cleaned_records = []

    required_columns = [
        "employee_id",
        "name",
        "age",
        "department",
        "designation",
        "salary",
        "location",
        "joining_date",
        "skills",
    ]

    # 1. Column structure check
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns in 'Employees' sheet: {', '.join(missing_cols)}")
        return False, errors, []

    if df.empty:
        errors.append("The 'Employees' sheet is empty.")
        return False, errors, []

    seen_ids = set()
    id_to_manager = {}
    row_cleaned = []

    # 2. Row-by-row field validation
    for idx, row in df.iterrows():
        row_num = idx + 2  # 1-indexed Excel row including header
        emp_id_raw = row.get("employee_id")
        emp_id = str(emp_id_raw).strip() if pd.notna(emp_id_raw) else ""

        if not emp_id or emp_id.lower() == "nan":
            errors.append(f"Row {row_num}: 'employee_id' is blank or missing.")
            continue

        if emp_id in seen_ids:
            errors.append(f"Row {row_num}: Duplicate employee_id '{emp_id}' found.")
        seen_ids.add(emp_id)

        # Name check
        name_raw = row.get("name")
        name = str(name_raw).strip() if pd.notna(name_raw) else ""
        if not name or name.lower() == "nan":
            errors.append(f"Row {row_num} ({emp_id}): 'name' cannot be blank.")

        # Age check
        age_raw = row.get("age")
        try:
            age = int(age_raw)
            if age <= 0 or age > 120:
                errors.append(f"Row {row_num} ({emp_id}): Invalid age '{age_raw}'. Must be between 1 and 120.")
        except (ValueError, TypeError):
            errors.append(f"Row {row_num} ({emp_id}): 'age' must be a valid integer, got '{age_raw}'.")
            age = None

        # Department check
        dept_raw = row.get("department")
        department = str(dept_raw).strip() if pd.notna(dept_raw) else ""
        if not department or department.lower() == "nan":
            errors.append(f"Row {row_num} ({emp_id}): 'department' cannot be blank.")

        # Designation check
        desig_raw = row.get("designation")
        designation = str(desig_raw).strip() if pd.notna(desig_raw) else ""
        if not designation or designation.lower() == "nan":
            errors.append(f"Row {row_num} ({emp_id}): 'designation' cannot be blank.")

        # Salary check
        salary_raw = row.get("salary")
        try:
            salary = float(salary_raw)
            if salary <= 0:
                errors.append(f"Row {row_num} ({emp_id}): 'salary' must be positive, got '{salary_raw}'.")
            salary = int(salary)
        except (ValueError, TypeError):
            errors.append(f"Row {row_num} ({emp_id}): 'salary' must be numeric, got '{salary_raw}'.")
            salary = None

        # Location check
        loc_raw = row.get("location")
        location = str(loc_raw).strip() if pd.notna(loc_raw) else ""
        if not location or location.lower() == "nan":
            errors.append(f"Row {row_num} ({emp_id}): 'location' cannot be blank.")

        # Joining date check
        jdate_raw = row.get("joining_date")
        if pd.isna(jdate_raw) or str(jdate_raw).strip() == "":
            errors.append(f"Row {row_num} ({emp_id}): 'joining_date' cannot be blank.")
            joining_date = ""
        else:
            try:
                joining_date = pd.to_datetime(jdate_raw).strftime("%Y-%m-%d")
            except Exception:
                errors.append(f"Row {row_num} ({emp_id}): Invalid 'joining_date' format '{jdate_raw}'.")
                joining_date = str(jdate_raw).strip()

        # Manager ID check
        mgr_raw = row.get("manager_id")
        manager_id = str(mgr_raw).strip() if pd.notna(mgr_raw) and str(mgr_raw).strip().lower() != "nan" else ""

        # Self-reporting check
        if manager_id and manager_id == emp_id:
            errors.append(f"Row {row_num} ({emp_id}): Self-reporting detected (manager_id equals employee_id).")

        id_to_manager[emp_id] = manager_id

        # Skills check
        skills_raw = row.get("skills")
        if pd.isna(skills_raw) or str(skills_raw).strip() == "":
            errors.append(f"Row {row_num} ({emp_id}): 'skills' cannot be blank.")
            skills = []
        else:
            skills = [s.strip() for s in str(skills_raw).split(",") if s.strip()]
            if not skills:
                errors.append(f"Row {row_num} ({emp_id}): 'skills' list must contain at least one valid skill.")

        row_cleaned.append({
            "employee_id": emp_id,
            "name": name,
            "age": age,
            "department": department,
            "designation": designation,
            "salary": salary,
            "location": location,
            "joining_date": joining_date,
            "manager_id": manager_id,
            "skills": skills,
            "row_num": row_num,
        })

    # 3. Manager existence validation
    for item in row_cleaned:
        mgr = item["manager_id"]
        if mgr and mgr not in seen_ids:
            errors.append(
                f"Row {item['row_num']} ({item['employee_id']}): Referenced manager_id '{mgr}' does not exist in employee dataset."
            )

    # 4. Circular reporting hierarchy cycle detection
    for emp_id in seen_ids:
        visited = []
        curr = emp_id
        while curr:
            if curr in visited:
                cycle_path = " -> ".join(visited + [curr])
                errors.append(f"Circular reporting relationship detected: {cycle_path}")
                break
            visited.append(curr)
            curr = id_to_manager.get(curr, "")

    if errors:
        return False, errors, []

    return True, [], row_cleaned


def create_constraints(session):
    """Create uniqueness constraints for Employee, Department, and Skill if supported."""
    constraints = [
        (
            "employee_id_unique",
            "CREATE CONSTRAINT employee_id_unique IF NOT EXISTS FOR (e:Employee) REQUIRE e.employee_id IS UNIQUE",
        ),
        (
            "department_name_unique",
            "CREATE CONSTRAINT department_name_unique IF NOT EXISTS FOR (d:Department) REQUIRE d.name IS UNIQUE",
        ),
        (
            "skill_name_unique",
            "CREATE CONSTRAINT skill_name_unique IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE",
        ),
    ]

    for name, cql in constraints:
        try:
            session.run(cql)
        except Exception:
            pass


def import_employee_data(cleaned_records, driver):
    """Import validated records into Neo4j inside transactions."""
    departments = sorted({r["department"] for r in cleaned_records})
    all_skills = sorted({s for r in cleaned_records for s in r["skills"]})

    with driver.session(database=DATABASE) as session:
        # Create schema constraints
        create_constraints(session)

        # 1. Merge Departments
        for dept in departments:
            session.run(
                "MERGE (d:Department {name: $dept})",
                dept=dept,
            )

        # 2. Merge Skills
        for skill in all_skills:
            session.run(
                "MERGE (s:Skill {name: $skill})",
                skill=skill,
            )

        # 3. Merge Employees and connect to Department and Skills
        emp_query = """
        MERGE (e:Employee {employee_id: $employee_id})
        SET e.name = $name,
            e.age = $age,
            e.designation = $designation,
            e.salary = $salary,
            e.location = $location,
            e.joining_date = $joining_date
        WITH e
        MATCH (d:Department {name: $department})
        MERGE (e)-[:WORKS_IN]->(d)
        WITH e
        UNWIND $skills AS skill_name
        MATCH (s:Skill {name: skill_name})
        MERGE (e)-[:HAS_SKILL]->(s)
        """
        for r in cleaned_records:
            session.run(
                emp_query,
                employee_id=r["employee_id"],
                name=r["name"],
                age=r["age"],
                designation=r["designation"],
                salary=r["salary"],
                location=r["location"],
                joining_date=r["joining_date"],
                department=r["department"],
                skills=r["skills"],
            )

        # 4. Merge REPORTS_TO relationships
        rel_count = 0
        mgr_query = """
        MATCH (e:Employee {employee_id: $employee_id})
        MATCH (m:Employee {employee_id: $manager_id})
        MERGE (e)-[:REPORTS_TO]->(m)
        """
        for r in cleaned_records:
            if r["manager_id"]:
                session.run(
                    mgr_query,
                    employee_id=r["employee_id"],
                    manager_id=r["manager_id"],
                )
                rel_count += 1

    return {
        "employees_count": len(cleaned_records),
        "departments_count": len(departments),
        "skills_count": len(all_skills),
        "manager_relationships": rel_count,
    }


def run_import(excel_path=None):
    """High-level runner: reads Excel, validates, and imports into Neo4j."""
    print("=" * 60)
    print("Neo4j Employee Knowledge Graph Ingestion")
    print("=" * 60)

    try:
        resolved_path = find_excel_file(excel_path)
        print(f"Reading workbook from: {resolved_path}")
    except FileNotFoundError as err:
        print(f"\n[ERROR] {err}")
        return False

    try:
        excel_file = pd.ExcelFile(resolved_path)
        if "Employees" not in excel_file.sheet_names:
            print("\n[ERROR] Workbook is missing the required 'Employees' sheet.")
            print(f"Available sheets: {', '.join(excel_file.sheet_names)}")
            return False

        df = pd.read_excel(excel_file, sheet_name="Employees")
    except Exception as err:
        print(f"\n[ERROR] Failed to read Excel file: {err}")
        return False

    print(f"Validating {len(df)} records in 'Employees' sheet...")
    is_valid, errors, cleaned_records = validate_employees_data(df)

    if not is_valid:
        print(f"\n[VALIDATION FAILED] Found {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        print("\nAborted import. No database records were modified.")
        return False

    print("Validation passed! All records, types, and reporting hierarchies are valid.")

    driver = None
    try:
        print("Connecting to Neo4j AuraDB...")
        driver = get_driver()
        driver.verify_connectivity()
        print("Connected successfully.")

        print("Importing nodes and relationships into Neo4j...")
        summary = import_employee_data(cleaned_records, driver)

        print("\n" + "=" * 60)
        print("IMPORT COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"  Employees imported/updated : {summary['employees_count']}")
        print(f"  Departments mapped         : {summary['departments_count']}")
        print(f"  Skills mapped              : {summary['skills_count']}")
        print(f"  Manager relationships (REPORTS_TO): {summary['manager_relationships']}")
        print("Existing movie recommendation data remains untouched.")
        return True

    except Exception as err:
        print(f"\n[DATABASE ERROR] Failed during import: {err}")
        return False

    finally:
        if driver is not None:
            driver.close()


def main():
    parser = argparse.ArgumentParser(description="Import employee Excel workbook into Neo4j.")
    parser.add_argument("--file", help="Path to employees.xlsx", default=None)
    args = parser.parse_args()

    success = run_import(args.file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
