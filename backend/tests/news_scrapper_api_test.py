import os
import sys
import asyncio
import httpx

# Ensure project root is in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

async def test_news_scraper_api():
    # Test configuration matching the ScraperConfig model
    test_config = {
        "user_id": "karim",
        "website_url": "https://www.jeuneafrique.com",
        "username": "Anas.abdoun@gmail.com",
        "password": "Kolxw007",
        "crawl": True,
        "max_pages": 3
    }

    # Use httpx for async HTTP requests to FastAPI
    async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout
        response = await client.post(
            "http://localhost:8000/news-scraper/scrape", 
            json=test_config,
            timeout=300.0
        )

    
    # Check response status
    assert response.status_code == 200, f"API request failed: {response.text}"

    # Parse response
    result = response.json()

    print(result)
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "articles" in result, "Response missing 'articles' key"
    assert "status" in result, "Response missing 'status' key"
    assert result["status"] == "success", "Scraping was not successful"

    # Validate articles structure
    articles = result["articles"]
    assert isinstance(articles, list), "Articles should be a list"
    assert len(articles) > 0, "Articles list is empty"

    # Validate article metrics
    article_metrics = articles[0]
    assert "total_urls_visited" in article_metrics, "Missing total_urls_visited metric"
    assert "article_count" in article_metrics, "Missing article_count metric"
    assert "non_article_count" in article_metrics, "Missing non_article_count metric"

if __name__ == "__main__":
    # For direct script execution
    asyncio.run(test_news_scraper_api())
