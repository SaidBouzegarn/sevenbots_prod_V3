from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, inspect, create_engine
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError
import json
import dotenv
import os
import pandas as pd
import asyncio
import asyncpg
from datetime import datetime
import logging
import sys


logger = logging.getLogger(__name__)
dotenv.load_dotenv()
def get_aws_secret(secret_name, region_name="eu-west-3"):
    """
    Retrieve database credentials from AWS Secrets Manager
    
    Args:
        secret_name (str): Name of the secret in AWS Secrets Manager
        region_name (str): AWS region where the secret is stored
    
    Returns:
        dict: Database connection credentials
    """
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)
    
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise e
    
    if 'SecretString' in get_secret_value_response:
        return json.loads(get_secret_value_response['SecretString'])
    
    raise ValueError("No secret found")

secret_name = os.getenv("AWS_SECRET_NAME")
rds_credentials = get_aws_secret(secret_name)

# RDS Connection Details
rds_host = "sevenbots.c9w2g0i8kg7w.eu-west-3.rds.amazonaws.com"
rds_port = "5432"

# Important: Use the default 'postgres' database to first connect and create the target database
default_dbname = "postgres"
target_dbname = "sevenbots"  # The database you want to create/use

class AsyncRDSPostgresManager:
    def __init__(self, 
                 host: str = rds_host, 
                 database: str = target_dbname, 
                 user: str = rds_credentials['username'], 
                 password: str = rds_credentials['password'], 
                 port: int = rds_port,
                 echo: bool = False):
        """
        Initialize the async RDS PostgreSQL database connection using SQLAlchemy.
        """
        # Construct SQLAlchemy async database URL
        self.db_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
        
        try:
            # Create async engine with connection pooling
            self.engine = create_async_engine(
                self.db_url, 
                echo=echo,  # Logging
                pool_pre_ping=True,  # Connection health check
                pool_recycle=3600  # Reconnect after 1 hour
            )
            
            # Create async session factory
            self.AsyncSessionLocal = async_sessionmaker(
                bind=self.engine, 
                class_=AsyncSession,
                expire_on_commit=False
            )
        except Exception as e:
            print(f"Error creating async database engine: {e}")
            raise

    async def get_connection(self) -> AsyncSession:
        """
        Get an async database session.
        
        :return: SQLAlchemy AsyncSession
        """
        # Create and return a new session directly
        return self.AsyncSessionLocal()

    async def get_tables(self):
        """
        Retrieve list of available tables in the database.
        """
        try:
            async with self.engine.connect() as conn:
                # Use raw connection to get table names
                result = await conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
                return [row[0] for row in result.fetchall()]
        except SQLAlchemyError as e:
            print(f"Error retrieving tables: {e}")
            raise

    async def query_table(self, 
                    table_name: str, 
                    conditions: Optional[Dict[str, Any]] = None, 
                    columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Async query a table with optional conditions and column selection.
        """
        session =  await self.get_connection()
            
        try:
            # Prepare base query using text constructs
            if columns:
                column_list = [text(f"{table_name}.{col}") for col in columns]
                query = text(f"SELECT {', '.join(str(col) for col in column_list)} FROM {table_name}")
            else:
                query = text(f"SELECT * FROM {table_name}")

            # Add conditions if provided
            if conditions:
                where_clauses = [f"{table_name}.{col} = :{col}" for col in conditions.keys()]
                if where_clauses:
                    query = text(f"{query} WHERE {' AND '.join(where_clauses)}")

            # Execute the query
            result = await session.execute(query, conditions or {})
            
            # Fetch the results
            rows = result.all()
            
            if not rows:
                return []

            # Get column names directly from the result description
            if columns:
                return [dict(zip(columns, row)) for row in rows]
            
            # If no specific columns were requested, use the column names from the result
            column_names = result.keys()
            return [dict(zip(column_names, row)) for row in rows]
        

    
        except SQLAlchemyError as e:
            print(f"Error querying table {table_name}: {e}")
            raise
        finally:
            await session.close()

    async def add_rows(self, table_name: str, data: List[Dict[str, Any]]):
        """
        Async add multiple rows to a specified table.
        
        Args:
            table_name (str): Name of the table to insert rows into
            data (List[Dict[str, Any]]): List of dictionaries, each representing a row to insert
        """
        if not data or not isinstance(data, list) or len(data) == 0:
            return
        
        session = await self.get_connection()
        try:
            # Get column names from the first dictionary in the input data
            columns = list(data[0].keys())
            
            # Construct INSERT statement for multiple rows
            placeholders = []
            for i in range(len(data)):
                row_placeholders = [f":{i}_{col}" for col in columns]
                placeholders.append(f"({', '.join(row_placeholders)})")
            
            insert_stmt = text(f"""
                INSERT INTO {table_name} ({', '.join(columns)}) 
                VALUES {', '.join(placeholders)}
                ON CONFLICT DO NOTHING
            """)
            
            # Prepare parameters for all rows
            params = {}
            for i, row in enumerate(data):
                for col, val in row.items():
                    params[f"{i}_{col}"] = val
            
            # Execute the insert for all rows at once
            await session.execute(insert_stmt, params)
            await session.commit()
            print(f"{len(data)} rows added to {table_name} successfully.")
        
        except SQLAlchemyError as e:
            await session.rollback()
            print(f"Error adding rows to {table_name}: {e}")
            raise
        finally:
            await session.close()

    async def delete_rows(self, table_name: str, conditions: Dict[str, Any]):
        """
        Async delete rows from a specified table based on conditions.
        
        Args:
            table_name (str): Name of the table to delete from
            conditions (Dict[str, Any]): Dictionary of column-value pairs to match for deletion
        """
        if not conditions or not isinstance(conditions, dict):
            raise ValueError("Conditions must be a non-empty dictionary")
        
        session = await self.get_connection()
        try:
            # Construct DELETE statement with conditions
            where_clauses = [f"{col} = :{col}" for col in conditions.keys()]
            delete_stmt = text(f"""
                DELETE FROM {table_name}
                WHERE {' AND '.join(where_clauses)}
            """)
            
            # Execute the delete statement with the conditions
            await session.execute(delete_stmt, conditions)
            await session.commit()
            
        except Exception as e:
            # Roll back and log the full exception details including stack trace
            await session.rollback()
            logger.exception(
                f"Error deleting rows from '{table_name}' "
                f"with conditions {conditions}: {e}"
            )
            raise
        finally:
            await session.close()

    async def update_rows(
        self, 
        table_name: str, 
        conditions: Dict[str, Any], 
        updated_values: Dict[str, Any]
    ) -> int:
        """
        Updates rows that match the specified conditions with the provided updated_values.
        
        :param table_name: Name of the table to update.
        :param conditions: Dictionary of column-value pairs for the WHERE clause.
        :param updated_values: Dictionary of column-value pairs to update.
        :return: The number of rows updated.
        """
        session = await self.get_connection()
        try:
            where_clauses = [f"{col} = :where_{col}" for col in conditions.keys()]
            set_clauses = [f"{col} = :update_{col}" for col in updated_values.keys()]
            
            update_stmt = text(f"""
                UPDATE {table_name}
                SET {', '.join(set_clauses)}
                WHERE {' AND '.join(where_clauses)}
            """)
            
            params = {}
            # Prefix condition bindings with "where_"
            for col, val in conditions.items():
                params[f"where_{col}"] = val
            # Prefix updated values with "update_"
            for col, val in updated_values.items():
                params[f"update_{col}"] = val

            result = await session.execute(update_stmt, params)
            await session.commit()

            return result.rowcount or 0   # rowcount holds the number of affected rows

        except SQLAlchemyError as e:
            await session.rollback()
            logger.exception(f"Error updating rows in '{table_name}': {e}")
            raise

    # Async context manager support
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Close engine if needed
        if hasattr(self, 'engine'):
            await self.engine.dispose()


