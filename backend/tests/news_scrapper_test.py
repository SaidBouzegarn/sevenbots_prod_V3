import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from backend.news_scrapper import NewsScrapper
import asyncio

async def main():
    # Test website configuration
    test_config = {
        'user_id': 'karim',
        'website_url': 'https://www.jeuneafrique.com',
        'username': 'Anas.abdoun@gmail.com',
        'password': 'Kolxw007',
        'crawl': True,
        'max_pages': 3  # Limit for testing
    }
    
    # Initialize scraper using the async create method
    scraper = await NewsScrapper.create(**test_config)
    
    try:
        # Initialize the async browser
        await scraper.initialize()
        
        print(f"Starting scrape of {test_config['website_url']}...")
        
        # Perform scraping
        articles = await scraper.scrape()
        
        # Print results
        print(f"\n response is {articles}:")

    
    finally:
        # Ensure browser is closed
        await scraper.close()

if __name__ == "__main__":
    asyncio.run(main())