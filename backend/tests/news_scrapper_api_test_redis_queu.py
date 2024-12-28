import os
import sys
import asyncio
import httpx
import time

# Ensure project root is in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'


async def test_news_scraper_api():
    print("\n=== Starting News Scraper API Test ===")
    start_total_time = time.time()
    
    # Test configuration matching the ScraperConfig model
    test_config = {
        "user_id": "karim",
        "website_url": "https://www.jeuneafrique.com",
        "username": "Anas.abdoun@gmail.com",
        "password": "Kolxw007",
        "crawl": True,
        "max_pages": 10
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        print("1. Initiating scraping task...")
        response = await client.post(
            "http://localhost:8000/news-scraper/scrape", 
            json=test_config
        )
        assert response.status_code == 200, f"API request failed: {response.text}"
        print(f"   ✓ Task initiated successfully")
        
        # Get the task ID
        task_data = response.json()
        assert "task_id" in task_data, "Response missing task_id"
        task_id = task_data["task_id"]
        print(f"   ✓ Received task ID: {task_id}")
        
        print("\n2. Polling for results...")
        poll_start_time = time.time()
        
        max_wait_time = 300  # 5 minutes
        start_time = time.time()
        
        while True:
            if time.time() - start_time > max_wait_time:
                raise TimeoutError("Task took too long to complete")
            
            status_response = await client.get(
                f"http://localhost:8000/news-scraper/status/{task_id}"
            )
            assert status_response.status_code == 200, "Status check failed"
            
            status_data = status_response.json()
            if status_data["status"] == "completed":
                result = status_data["result"]
                poll_duration = time.time() - poll_start_time
                print(f"   ✓ Task completed in {poll_duration:.2f} seconds")
                break
            elif status_data["status"] == "failed":
                print(f"   ✗ Task failed: {status_data.get('error')}")
                raise Exception(f"Task failed: {status_data.get('error')}")
            
            print(f"   • Status: {status_data['status']} - waiting 2 seconds...")
            await asyncio.sleep(2)
        
        print("\n3. Validating results...")
        # Validate the result structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "total_urls_visited" in result, "Missing total_urls_visited metric"
        assert "article_count" in result, "Missing article_count metric"
        assert "non_article_count" in result, "Missing non_article_count metric"
        
        print(f"   ✓ Results validated successfully")
        print(f"   • Total URLs visited: {result['total_urls_visited']}")
        print(f"   • Articles found: {result['article_count']}")
        print(f"   • Non-articles found: {result['non_article_count']}")
        
        print("\n4. Testing task cancellation...")
        # Optional: Test task cancellation
        # Only if you want to test the cancel endpoint
        cancel_response = await client.delete(
            f"http://localhost:8000/news-scraper/tasks/{task_id}"
        )
        assert cancel_response.status_code == 200, "Task cancellation failed"
        print("   ✓ Task cancellation successful")

        total_duration = time.time() - start_total_time
        print(f"\n=== Test completed in {total_duration:.2f} seconds ===\n")

if __name__ == "__main__":
    asyncio.run(test_news_scraper_api())
