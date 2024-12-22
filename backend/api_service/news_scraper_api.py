from fastapi import APIRouter, HTTPException
from redis import Redis
from rq import Queue
from rq.job import Job
from typing import Optional
from pydantic import BaseModel
from backend.news_scrapper import NewsScrapper
from rq.worker import SimpleWorker

# Initialize Redis and RQ with SimpleWorker
redis_conn = Redis(host='localhost', port=6379)
scraper_queue = Queue('scraper', connection=redis_conn, worker_class=SimpleWorker)

news_scraper_router = APIRouter(prefix="/news-scraper", tags=["news-scraper"])


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
    task_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None

async def scrape_website_task(config: dict):
    try:
        scraper = await NewsScrapper.create(**config)
        await scraper.initialize()
        result = await scraper.scrape()
        
        total_urls, article_count, non_article_count = result
        return {
            "total_urls_visited": total_urls,
            "article_count": article_count,
            "non_article_count": non_article_count
        }
    finally:
        await scraper.close()

@news_scraper_router.post("/scrape", response_model=ScraperResponse)
async def start_scraping(config: ScraperConfig):
    # Enqueue the scraping task
    job = scraper_queue.enqueue(
        scrape_website_task,
        config.model_dump(),
        job_timeout='1h',  # Set maximum job runtime
        result_ttl=86400,  # Keep results for 24 hours
        failure_ttl=86400  # Keep failed job info for 24 hours
    )
    
    return ScraperResponse(
        task_id=job.id,
        status=job.get_status()
    )

@news_scraper_router.get("/status/{task_id}", response_model=ScraperResponse)
async def get_scraping_status(task_id: str):
    try:
        job = Job.fetch(task_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if job.is_finished:
        return ScraperResponse(
            task_id=task_id,
            status="completed",
            result=job.result
        )
    elif job.is_failed:
        return ScraperResponse(
            task_id=task_id,
            status="failed",
            error=str(job.exc_info)
        )
    
    return ScraperResponse(
        task_id=task_id,
        status=job.get_status()
    )

@news_scraper_router.delete("/tasks/{task_id}")
async def cancel_scraping(task_id: str):
    try:
        job = Job.fetch(task_id, connection=redis_conn)
        if job.get_status() != 'finished':
            job.cancel()
            job.delete()
            return {"message": "Task cancelled successfully"}
        return {"message": "Task already completed"}
    except Exception:
        raise HTTPException(status_code=404, detail="Task not found")

