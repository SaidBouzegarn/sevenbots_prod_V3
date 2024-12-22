from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from backend.db_manager import AsyncRDSPostgresManager

# Create a router for database management endpoints
db_manager_router = APIRouter(
    prefix="/db",
    tags=["Database Management"]
)

# Create a dependency to get a database manager instance
async def get_db_manager():
    """
    Dependency to create and manage database connection
    """
    db_manager = AsyncRDSPostgresManager()
    try:
        yield db_manager
    finally:
        # Ensure the engine is disposed after use
        if hasattr(db_manager, 'engine'):
            await db_manager.engine.dispose()


@db_manager_router.get("/tables", response_model=List[str])
async def list_tables(db_manager: AsyncRDSPostgresManager = Depends(get_db_manager)):
    """
    Endpoint to retrieve list of tables in the database
    """
    try:
        tables = await db_manager.get_tables()
        return tables
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving tables: {str(e)}")

@db_manager_router.post("/query/{table_name}")
async def query_table(
    table_name: str, 
    conditions: Optional[Dict[str, Any]] = None, 
    columns: Optional[List[str]] = None,
    db_manager: AsyncRDSPostgresManager = Depends(get_db_manager)
):
    """
    Endpoint to query a specific table with optional conditions and column selection
    """
    try:
        results = await db_manager.query_table(
            table_name, 
            conditions=conditions, 
            columns=columns
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying table {table_name}: {str(e)}")


@db_manager_router.post("/add_rows/{table_name}")
async def add_rows(
    table_name: str,
    data: List[Dict[str, Any]] = Body(...),
    db_manager: AsyncRDSPostgresManager = Depends(get_db_manager)
):
    """
    Endpoint to add multiple rows to a specified table
    """
    try:
        await db_manager.add_rows(table_name, data)
        return {"status": "success", "message": f"Rows added to {table_name}"}
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error adding rows to {table_name}: {str(e)}"
        )

@db_manager_router.post("/delete_rows/{table_name}")
async def delete_rows(
    table_name: str,
    conditions: Dict[str, Any] = Body(...),  # Use Body to explicitly handle raw JSON
    db_manager: AsyncRDSPostgresManager = Depends(get_db_manager)
):
    """
    Endpoint to delete rows from a specified table based on conditions
    """
    try:
        if not conditions:
            raise HTTPException(status_code=400, detail="Conditions cannot be empty")
            
        deleted_count = await db_manager.delete_rows(table_name, conditions)
        return {
            "status": "success", 
            "message": f"Rows deleted from {table_name}", 
            "deleted_count": deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting rows from {table_name}: {str(e)}")

@db_manager_router.post("/echo")
async def echo_data(payload: List[Dict[str, Any]] = Body(...)):
    """
    Test-only endpoint to receive and return a list of dicts exactly as sent
    """
    print("this is testing the payload transfer via fastapi change", payload)
    return payload

@db_manager_router.post("/update_rows/{table_name}")
async def update_table_rows(
    table_name: str,
    payload: Dict[str, Any] = Body(...),
    db_manager: AsyncRDSPostgresManager = Depends(get_db_manager)
):
    """
    Endpoint to update rows in the specified table based on given conditions and update values.

    Expected JSON body example:
    {
        "conditions": {"user_id": 123},
        "updated_values": {"name": "Alice", "status": "active"}
    }
    """
    # Extract conditions and updated_values from the request payload
    conditions = payload.get("conditions", {})
    updated_values = payload.get("updated_values", {})

    # Basic validation
    if not conditions or not updated_values:
        raise HTTPException(
            status_code=400, 
            detail="Both 'conditions' and 'updated_values' must be present and non-empty"
        )

    try:
        updated_count = await db_manager.update_rows(table_name, conditions, updated_values)
        return {
            "status": "success",
            "message": f"Rows updated in {table_name}",
            "updated_count": updated_count
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error updating rows in {table_name}: {str(e)}"
        )