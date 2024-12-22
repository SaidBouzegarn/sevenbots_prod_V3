import traceback 
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from backend.db_manager import AsyncRDSPostgresManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a router for databases-related endpoints
databases_router = APIRouter(prefix="/databases", tags=["databases"])

# Pydantic models for data validation
class TableQueryParams(BaseModel):
    table_name: str
    conditions: Optional[Dict[str, Any]] = None
    columns: Optional[List[str]] = None

class TableAddRowsParams(BaseModel):
    table_name: str
    rows: List[Dict[str, Any]]

async def get_db_manager():
    """
    Async dependency that yields a DB manager so FastAPI can
    correctly handle enter/exit of the async context manager.
    """
    return AsyncRDSPostgresManager()


@databases_router.post("/tables")
async def get_available_tables(db: AsyncRDSPostgresManager = Depends(get_db_manager)):
    """
    Get list of available tables asynchronously
    """
    try:
        tables = await db.get_tables()
        
        # Remove system tables and unnecessary tables
        excluded_tables = [
            'pg_catalog', 
            'information_schema', 
            # Add any other tables you want to exclude
        ]
        
        # Filter out excluded tables and system schemas
        filtered_tables = [
            table for table in tables 
            if not table in excluded_tables
        ]
        
        return filtered_tables
    except Exception as e:
        logger.error(f"Error retrieving tables: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error retrieving tables: {str(e)}")

@databases_router.post("/query")
async def query_table(
    params: TableQueryParams, 
    db: AsyncRDSPostgresManager = Depends(get_db_manager)
):
    """
    Async query table using existing query_table method
    """
    try:
        # Log the incoming query parameters for debugging
        logger.info(f"Querying table: {params.table_name}")
        logger.info(f"Conditions: {params.conditions}")
        logger.info(f"Columns: {params.columns}")

        # Use the injected db instance from the dependency 
        results = await db.query_table(
            table_name=params.table_name, 
            conditions=params.conditions, 
            columns=params.columns
        )

        return results
    
    except Exception as e:
        # Log the full traceback
        logger.error(f"Error querying table: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error querying table: {str(e)}")

@databases_router.post("/add_rows")
async def add_rows(
    params: TableAddRowsParams, 
    db: AsyncRDSPostgresManager = Depends(get_db_manager)
):
    """
    Async add multiple rows to a table
    """
    try:
        # Log the incoming parameters for debugging
        logger.info(f"Adding rows to table: {params.table_name}")
        logger.info(f"Number of rows: {len(params.rows)}")

        # Create a new database manager instance

        # Use the existing add_rows method from db_manager
        await db.add_rows(
            table_name=params.table_name, 
            data=params.rows
        )
        
        return {
            "status": "success", 
            "added_rows": len(params.rows)
        }
    
    except Exception as e:
        # Log the full traceback
        logger.error(f"Error adding rows: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error adding rows: {str(e)}")
