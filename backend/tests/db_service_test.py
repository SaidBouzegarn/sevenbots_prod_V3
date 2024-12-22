import os
import sys
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import asyncio
import httpx
import json
from datetime import datetime

# Base URL for the API
BASE_URL = "http://localhost:8000"  # Adjust this to match your FastAPI server's address



async def test_get_available_tables():
    """
    Test the /db/tables endpoint
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/db/tables")
        
        assert response.status_code == 200, f"Failed to get tables. Status: {response.status_code}"
        tables = response.json()
        
        print("Available Tables:")
        for table in tables:
            print(f"- {table}")
        
        assert len(tables) > 0, "No tables returned"
        assert isinstance(tables, list), "Tables should be returned as a list"

async def test_query_table():
    """
    Test the /db/query/{table_name} endpoint
    """
    async with httpx.AsyncClient() as client:
        test_table = "websites"
        query_payload = {}
        
        response = await client.post(
            f"{BASE_URL}/db/query/{test_table}", 
            json=query_payload
        )
        
        assert response.status_code == 200, f"Query failed. Status: {response.status_code}"
        results = response.json()

        print(f"Query Results for {test_table}:")
        print(json.dumps(results, indent=2))
        
        assert isinstance(results, list), "Results should be a list"
        assert len(results) > 0, "No results returned from table query"
        
        if results:
            first_result = results[0]
            assert isinstance(first_result, dict), "Each result should be a dictionary"

async def test_add_rows_endpoint():
    """
    Test the /db/add_rows/visited_urls endpoint
    """
    async with httpx.AsyncClient() as client:
        test_data = [
            {
                "user_id": "test_user",
                "domain": "example.com",
                "url": "https://example.com/article3",
                "is_article": True,
            },
            {
                "user_id": "test_user",
                "domain": "example.com",
                "url": "https://example.com/article4",
                "is_article": False,
            }
        ]

        response = await client.post(
            f"{BASE_URL}/db/add_rows/visited_urls",
            content=json.dumps(test_data),
            headers={"Content-Type": "application/json"}
        )

        print(response.json())

        assert response.status_code == 200, f"Failed to add rows. Status: {response.status_code}, Details: {response.text}"
        result = response.json()
        assert result["status"] == "success", "Response should indicate success"
        
        query_response = await client.post(
            f"{BASE_URL}/db/query/visited_urls",
            json={
                "conditions": {
                    "user_id": "test_user",
                    "domain": "example.com"
                }
            }
        )
        
        assert query_response.status_code == 200, "Failed to query added rows"
        results = query_response.json()
        print(results)
        
        assert len(results) > 0, "No results found for added test data"
        assert results[0]["url"] == "https://example.com/article3", "Added URL doesn't match test data"

async def test_delete_rows_endpoint():
    """
    Test the /db/delete_rows/visited_urls endpoint after adding test data
    """
    async with httpx.AsyncClient() as client:

        delete_conditions = {
            "user_id": "test_user",
            "domain": "example.com"
        }
        delete_response = await client.post(
            f"{BASE_URL}/db/delete_rows/visited_urls",
            content=json.dumps(delete_conditions),
            headers={"Content-Type": "application/json"}
        )
        print(delete_response.json())
        if delete_response.status_code != 200:
            print("Delete Error Response Status:", delete_response.status_code)
            print("Delete Error Response Content:", delete_response.text)
            print("Delete Request Data:", delete_conditions)
        assert delete_response.status_code == 200, "Failed to delete rows"
        
        verify_response = await client.post(
            f"{BASE_URL}/db/query/visited_urls",
            json={
                "conditions": delete_conditions
            }
        )
        assert verify_response.status_code == 200, "Failed to verify deletion"
        final_results = verify_response.json()
        assert len(final_results) == 0, "Rows were not properly deleted"

async def test_echo_data():
    """
    Test the /db/echo endpoint to ensure it can receive list[dict] 
    and return it exactly as sent.
    """
    async with httpx.AsyncClient() as client:
        # Prepare some sample list-of-dicts
        test_data = [
            {"key1": "value1", "another_key": 123},
            {"key2": "value2", "flag": True},
        ]

        # Send data using 'json=' so it’s properly encoded as JSON
        response = await client.post(
            f"{BASE_URL}/db/echo",
            json=test_data
        )

        assert response.status_code == 200, f"Echo endpoint failed. Status: {response.status_code}, Details: {response.text}"
        
        response_data = response.json()
        print("Echo Response:", response_data)

        # Verify the endpoint returned exactly what we sent
        assert response_data == test_data, "Echo response did not match the input payload!"

async def test_update_rows_endpoint():
    """
    Test the /db/update_rows/{table_name} endpoint.
    """
    async with httpx.AsyncClient() as client:
        
        # 1. First, add data we can update
        insert_data = [
            {
                "user_id": "test_user_update",
                "domain": "example.com",
                "url": "https://example.com/update-me1",
                "is_article": True
            }
        ]
        add_response = await client.post(
            f"{BASE_URL}/db/add_rows/visited_urls",
            content=json.dumps(insert_data), 
            headers={"Content-Type": "application/json"}
        )
        
        assert add_response.status_code == 200, (
            f"Failed to add rows before update test. "
            f"Status: {add_response.status_code}, Details: {add_response.text}"
        )
        
        # 2. Prepare the update payload
        update_payload = {
            "conditions": {
                "user_id": "test_user_update",
                "url": "https://example.com/update-me1"
            },
            "updated_values": {
                "is_article": False
            }
        }
        
        # 3. Send the update request
        update_response = await client.post(
            f"{BASE_URL}/db/update_rows/visited_urls",
            content=json.dumps(update_payload),
            headers={"Content-Type": "application/json"}
        )
        
        assert update_response.status_code == 200, (
            f"Update rows endpoint failed. "
            f"Status: {update_response.status_code}, Details: {update_response.text}"
        )
        
        update_result = update_response.json()
        print("Update Rows Response:", update_result)
        assert update_result["status"] == "success", "Update response should indicate success."
        assert update_result["updated_count"] == 1, "Exactly one row should have been updated."
        
        # 4. Query to verify the update
        query_payload = {
            "conditions": {
                "user_id": "test_user_update"
            }
        }
        query_response = await client.post(
            f"{BASE_URL}/db/query/visited_urls",
            content=json.dumps(query_payload),
            headers={"Content-Type": "application/json"}
        )
        assert query_response.status_code == 200, \
            f"Failed to query rows after update. Status: {query_response.status_code}"
        
        query_results = query_response.json()
        print("Query Results After Update:", json.dumps(query_results, indent=2))
        
        updated_record = next((row for row in query_results if row["url"] == "https://example.com/update-me1"), None)
        assert updated_record is not None, "Expected updated record not found in query results."
        assert updated_record["is_article"] is False, "Record was not updated to the new value."
        
        # 5. Clean up by deleting the test data
        delete_payload = {
            "user_id": "test_user_update",
            "domain": "example.com"
        }
        delete_response = await client.post(
            f"{BASE_URL}/db/delete_rows/visited_urls",
            content=json.dumps(delete_payload),
            headers={"Content-Type": "application/json"}
        )
        assert delete_response.status_code == 200, \
            f"Failed to delete test rows. Status: {delete_response.status_code}"

async def main():
    """
    Run all endpoint-based database tests
    """
    print("Starting Endpoint-Based Database Service Tests...")
    
    # You can call existing tests here if desired:
    # await test_get_available_tables()
    # await test_query_table()
    # await test_add_rows_endpoint()
    # await test_delete_rows_endpoint()
    
    # Now call your new update test:
    await test_update_rows_endpoint()
    
    print("All Endpoint-Based Database Service Tests Completed!")

if __name__ == "__main__":
    asyncio.run(main())
