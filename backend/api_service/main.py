import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api_service.news_scraper_api import news_scraper_router
from backend.api_service.db_manager_api import db_manager_router


load_dotenv()

app = FastAPI(
    title="SevenBots API",
    description="API for SevenBots Backend",
    version="1.0.0",
    reload=True,
    timeout=300  # 5 minute timeout

)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(news_scraper_router)
app.include_router(db_manager_router)


