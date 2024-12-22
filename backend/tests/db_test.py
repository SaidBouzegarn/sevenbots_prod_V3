import os
import sys
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import asyncio
import json
from datetime import datetime
from fastapi.encoders import jsonable_encoder
from backend.db_manager import AsyncRDSPostgresManager

async def test_db_manager_get_tables():
    """
    Test getting tables directly from AsyncRDSPostgresManager
    """
    async with AsyncRDSPostgresManager() as db_manager:
        tables = await db_manager.get_tables()
        
        print("Tables from DB Manager:")
        for table in tables:
            print(f"- {table}")
        
        assert len(tables) > 0, "No tables found in database"
        assert isinstance(tables, list), "Tables should be returned as a list"

async def test_db_manager_query_table():
    """
    Test querying a table directly using AsyncRDSPostgresManager
    """
    async with AsyncRDSPostgresManager() as db_manager:
        results = await db_manager.query_table("websites")
        serializable_results = jsonable_encoder(results)
        
        print("Query Results:")
        print(json.dumps(serializable_results, indent=2))
        
        assert len(serializable_results) > 0, "No results returned from table query"
        assert isinstance(serializable_results, list), "Results should be a list of dictionaries"

        column_results = await db_manager.query_table("websites")
        serializable_column_results = jsonable_encoder(column_results)
        print("Column Results:")
        print(json.dumps(serializable_column_results, indent=2))

async def test_db_manager_add_rows():
    """
    Test adding rows directly using AsyncRDSPostgresManager
    """
    async with AsyncRDSPostgresManager() as db_manager:
        test_data = [
            {
                "user_id": "test_user",
                "domain": "example.com",
                "url": "https://example.com/page1",
                "is_article": True,
            },
            {
                "user_id": "test_user",
                "domain": "example.com",
                "url": "https://example.com/page2",
                "is_article": False,
            }
        ]
        
        # Add rows directly
        await db_manager.add_rows("visited_urls", test_data)
        
        # Verify the rows were added
        results = await db_manager.query_table(
            "visited_urls",
            conditions={
                "user_id": "test_user",
                "domain": "example.com"
            }
        )
        
        print("Add Rows Results:")
        print(json.dumps(jsonable_encoder(results), indent=2))
        
        assert all(row["domain"] == "example.com" for row in results), \
            "Test rows not inserted or domain mismatch"

async def test_db_manager_delete_rows():
    """
    Test deleting rows directly using AsyncRDSPostgresManager
    """
    async with AsyncRDSPostgresManager() as db_manager:
        # Use the same test data as above
        delete_test_data = [
            {
                "user_id": "test_user",
                "domain": "example.com",
                "url": "https://example.com/page1",
                "is_article": True,
            },
            {
                "user_id": "test_user",
                "domain": "example.com",
                "url": "https://example.com/page2",
                "is_article": False,
            }
        ]
        # Perform the deletion for exactly those rows
        for row in delete_test_data:
            await db_manager.delete_rows(
                "visited_urls",
                conditions={
                    "user_id": row["user_id"],
                    "domain": row["domain"],
                    "url": row["url"]
                }
            )

        # Confirm they are gone
        post_delete_results = await db_manager.query_table(
            "visited_urls",
            conditions={
                "user_id": "test_user",
                "domain": "example.com"
            }
        )
        #assert len(post_delete_results) == 0, "Should have been deleted but still found"

async def test_db_manager_update_rows():
    """
    Test updating rows directly using AsyncRDSPostgresManager
    """
    async with AsyncRDSPostgresManager() as db_manager:
        # 1. Insert some rows to update later
        insert_data = [
            {
                "user_id": "test_user_update",
                "domain": "example.com",
                "url": "https://example.com/update-me1",
                "is_article": True,
            },
            {
                "user_id": "test_user_update",
                "domain": "example.com",
                "url": "https://example.com/update-me2",
                "is_article": False,
            }
        ]
        await db_manager.add_rows("visited_urls", insert_data)

        # 2. Update a row
        conditions = {
            "user_id": "test_user_update",
            "url": "https://example.com/update-me1"
        }
        updated_values = {
            "is_article": False
        }
        updated_count = await db_manager.update_rows("visited_urls", conditions, updated_values)
        print(f"Updated {updated_count} rows.")

        assert updated_count == 1, "Exactly one row should have been updated"

        # 3. Verify the row was updated
        results = await db_manager.query_table(
            "visited_urls",
            conditions={"user_id": "test_user_update"}
        )

        # For demonstration, print out the rows
        print("Update Rows Results:")
        print(json.dumps(jsonable_encoder(results), indent=2))

        # Ensure the specific row was updated
        updated_record = next(
            (row for row in results if row["url"] == "https://example.com/update-me1"), 
            None
        )
        assert updated_record is not None, "Updated record not found in query results"
        assert updated_record["is_article"] == False, "Row did not have the updated value"

        # 4. Clean up so it doesn’t affect other tests
        await db_manager.delete_rows(
            "visited_urls",
            conditions={"user_id": "test_user_update"}
        )

async def main():
    """
    Run all DB Manager Direct Method Tests
    """
    print("Starting DB Manager Tests...")
    
    await test_db_manager_get_tables()
    await test_db_manager_query_table()
    await test_db_manager_add_rows()
    await test_db_manager_delete_rows()
    await test_db_manager_update_rows()
    
    print("All DB Manager Tests Completed!")

if __name__ == "__main__":
    asyncio.run(main())
