from fastapi import (
    APIRouter, 
    HTTPException, 
    BackgroundTasks,
    Depends
)
from pydantic import BaseModel
from typing import Optional, List
from backend.news_scrapper import NewsScrapper
import logging
import asyncio
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

class ScraperConfig(BaseModel):
    user_id: str
    website_url: str
    username: Optional[str] = None
    password: Optional[str] = None
    login_url: Optional[str] = None
    username_selector: Optional[str] = None
    password_selector: Optional[str] = None
    submit_button_selector: Optional[str] = None
    login_url_ok: bool = False
    username_selector_ok: bool = False
    password_selector_ok: bool = False
    submit_button_selector_ok: bool = False
    crawl: bool = True
    max_pages: int = 25

class ScraperResponse(BaseModel):
    articles: List[dict]
    status: str

news_scraper_router = APIRouter(prefix="/news-scraper", tags=["news-scraper"])

@news_scraper_router.post("/scrape", response_model=ScraperResponse)
async def scrape_website(config: ScraperConfig):
    # Convert Pydantic model to dictionary for NewsScrapper
    scraper_config = config.model_dump()
    
    try:
        # Initialize scraper
        scraper = await NewsScrapper.create(**scraper_config)
        
        await scraper.initialize()
        # Perform scraping asynchronously
        result = await scraper.scrape()
        
        # Convert the tuple result into the expected format
        total_urls, article_count, non_article_count = result
        return {
            "articles": [
                {
                    "total_urls_visited": total_urls,
                    "article_count": article_count,
                    "non_article_count": non_article_count
                }
            ],
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Scraping error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await scraper.close()
    

