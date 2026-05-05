import json
import sqlite3
from typing import Any
from langchain_core.tools import tool
from app.browser_promoter.database import initialize_persistence_database, provision_dynamic_table, log_data_access

@tool
def create_persistent_table(table_name: str, schema_json_str: str) -> str:
    """
    Dynamically provision a new SQLite table.
    table_name (str): Name of the table. Prefix 'dynamic_' will be automatically added if missing.
    schema_json_str (str): JSON string mapping column_name -> sqlite_data_type (e.g., '{"id": "INTEGER PRIMARY KEY", "data": "TEXT"}')
    """
    try:
        schema = json.loads(schema_json_str)
        provision_dynamic_table(table_name, schema)
        log_data_access("db_tools", "CREATE", table_name, "Table provisioned")
        return f"Table {table_name} provisioned successfully."
    except Exception as e:
        return f"Error provisioning table: {e}"

@tool
def query_dynamic_table(query: str) -> str:
    """
    Execute a read (SELECT) query on the database.
    query (str): The SQL query.
    Returns the JSON representation of the rows.
    """
    try:
        db_path = initialize_persistence_database()
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query)
            rows = cursor.fetchall()
            log_data_access("db_tools", "READ", "various", query)
            return json.dumps([dict(row) for row in rows])
    except Exception as e:
        return f"Error executing query: {e}"

@tool
def execute_dynamic_table_write(query: str, parameters_json_str: str = "[]") -> str:
    """
    Execute a write (INSERT/UPDATE/DELETE) query on the database.
    query (str): The SQL query with ? placeholders.
    parameters_json_str (str): JSON list of parameters.
    """
    try:
        parameters = json.loads(parameters_json_str)
        db_path = initialize_persistence_database()
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(query, parameters)
            conn.commit()
            log_data_access("db_tools", "WRITE", "various", query)
            return f"Query executed successfully. Rows affected: {cursor.rowcount}"
    except Exception as e:
        return f"Error executing write query: {e}"
