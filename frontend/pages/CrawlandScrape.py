import sys
import os
import asyncio
import aiohttp
import streamlit as st
import pandas as pd
import subprocess
import logging
import requests
import time

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from backend.utils.logging_config import setup_cloudwatch_logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

async def install_playwright_browsers():
    """
    Asynchronously installs Playwright browsers and their dependencies if they are not already installed.
    """
    try:
        logger.info("Installing Playwright browsers...")
        process = await asyncio.create_subprocess_exec(
            "playwright", "install",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            logger.info("Playwright browsers installed successfully.")
        else:
            logger.error(f"Failed to install Playwright: {stderr.decode()}")
            st.error("Failed to install Playwright browsers or dependencies. Please check the logs for more details.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error installing Playwright: {str(e)}")
        st.error(f"Failed to install Playwright: {str(e)}")
        sys.exit(1)

async def scrape_website_async(session, config):
    """
    Asynchronous function to scrape a website using the FastAPI news scraper service
    """
    try:
        # Start the scraping task
        async with session.post(
            "http://localhost:8000/news-scraper/scrape", 
            json=config
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                st.error(f"Failed to start scraping for {config['website_url']}: {error_text}")
                return None
            
            task_data = await response.json()
            task_id = task_data.get('task_id')
            
            if not task_id:
                st.error(f"No task ID received for {config['website_url']}")
                return None
            
            # Poll for results
            max_wait_time = 300  # 5 minutes
            start_time = time.time()
            
            while True:
                if time.time() - start_time > max_wait_time:
                    st.error(f"Timeout waiting for results from {config['website_url']}")
                    return None
                
                async with session.get(
                    f"http://localhost:8000/news-scraper/status/{task_id}"
                ) as status_response:
                    if status_response.status != 200:
                        st.error(f"Failed to get status for {config['website_url']}")
                        return None
                    
                    status_data = await status_response.json()
                    
                    if status_data["status"] == "completed":
                        return status_data["result"]
                    elif status_data["status"] == "failed":
                        st.error(f"Scraping failed for {config['website_url']}: {status_data.get('error')}")
                        return None
                    
                    # Wait before next poll
                    await asyncio.sleep(2)
                    
    except Exception as e:
        st.error(f"Error scraping {config['website_url']}: {str(e)}")
        return None

async def fetch_user_websites():
    """Fetch websites for the current user"""
    try:
        # Ensure user is logged in
        if not st.session_state.get("username_id"):
            st.warning("Please log in to view websites")
            return []
        
        # Prepare the API call to fetch websites
        api_url = f"{st.secrets.get('API_BASE_URL', 'http://localhost:8000')}/db/query/websites"
        
        # Conditions to filter by user_id
        conditions = {"user_id": st.session_state.username_id}
        
        # Columns to exclude user_id if needed
        columns = [
            'url', 'login_url', 'username', 'password', 
            'username_selector', 'password_selector', 'submit_button_selector',
            'login_url_ok', 'username_selector_ok', 'password_selector_ok', 
            'submit_button_selector_ok', 'logged_in'
        ]
        
        # Make the API request synchronously since we're in an async context
        response = requests.post(
            api_url, 
            json={
                "conditions": conditions,
                "columns": columns
            }
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch websites: {response.text}")
            return []
    
    except Exception as e:
        st.error(f"Error fetching websites: {e}")
        return []

def ensure_valid_url(url: str) -> str:
    """Ensure URL has a proper protocol prefix"""
    if url and not url.startswith(('http://', 'https://')):
        return f'https://{url}'
    return url

async def scrape_selected_websites(session, selected_websites):
    """
    Scrape selected websites asynchronously
    """
    all_scraped_articles = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, website in enumerate(selected_websites):
        # Ensure all config values are strings or None and URLs are properly formatted
        config = {
            'user_id': str(st.session_state.username_id),
            'website_url': ensure_valid_url(str(website.get('url', '')).strip()),
            'login_url': ensure_valid_url(str(website.get('login_url', '')).strip()) if website.get('login_url') else None,
            'username': str(website.get('username', '')).strip() if website.get('username') else None,
            'password': str(website.get('password', '')).strip() if website.get('password') else None,
            'username_selector': str(website.get('username_selector', '')).strip() if website.get('username_selector') else None,
            'password_selector': str(website.get('password_selector', '')).strip() if website.get('password_selector') else None,
            'submit_button_selector': str(website.get('submit_button_selector', '')).strip() if website.get('submit_button_selector') else None,
            'login_url_ok': bool(website.get('login_url_ok', False)),
            'username_selector_ok': bool(website.get('username_selector_ok', False)),
            'password_selector_ok': bool(website.get('password_selector_ok', False)),
            'submit_button_selector_ok': bool(website.get('submit_button_selector_ok', False)),
            'crawl': True,
            #'max_pages': int(website.get('max_pages', 25)),
            'max_pages': 10
        }
        
        # Validate required URLs
        if not config['website_url']:
            st.error(f"Invalid website URL for entry {i+1}")
            continue
            
        # Update status
        status_text.text(f"Scraping {config['website_url']} ({i+1}/{len(selected_websites)})")
        
        # Log the config for debugging
        logger.info(f"Scraping config for {config['website_url']}: {config}")
        
        # Perform scraping
        result = await scrape_website_async(session, config)
        
        if result:
            # Update progress
            progress = (i + 1) / len(selected_websites)
            progress_bar.progress(progress)
            
            st.success(f"Successfully scraped {config['website_url']}")
            
            # Add results to collection
            all_scraped_articles.append({
                'website': config['website_url'],
                'total_urls': result.get('total_urls_visited', 0),
                'articles': result.get('article_count', 0),
                'non_articles': result.get('non_article_count', 0)
            })
        else:
            st.warning(f"No results for {config['website_url']}")
        
    status_text.empty()
    return all_scraped_articles

async def render_crawl_page():
    logger.info("Rendering crawl page")
    st.title("Website Crawler")
    
    # Ensure Playwright browsers and dependencies are installed
    await install_playwright_browsers()
    
    # Ensure user_id is set in session state or quit and ask user to log in
    if 'username_id' not in st.session_state:
        st.error("User ID not found. Please log in and try again.")
        st.stop()   

    # Split the screen into two columns
    left_col, right_col = st.columns(2)
    
    # Left column - Website List
    with left_col:
        st.header("Your Websites")
        
        # Fetch user's websites
        try:
            user_websites = await fetch_user_websites()
            
            if not user_websites:
                st.warning("No websites found. Add websites to your account.")
                return
            
            # Create a DataFrame for websites
            df = pd.DataFrame(user_websites)
            
            # Multiselect for websites
            selected_website_indices = st.multiselect(
                "Select websites to scrape", 
                options=range(len(user_websites)),
                format_func=lambda x: user_websites[x].get('url', 'Unknown URL')
            )
            
            # Display website details
            if selected_website_indices:
                selected_websites_details = [user_websites[i] for i in selected_website_indices]
                st.write("Selected Websites:")
                for website in selected_websites_details:
                    st.text(f"- {website.get('url', 'Unknown URL')}")
        
        except Exception as e:
            st.error(f"Error loading websites: {str(e)}")
            return
    
    # Right column - Scrape Buttons
    with right_col:
        st.header("Scrape Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scrape Selected Websites Button
            if st.button("Scrape Selected"):
                if selected_website_indices:
                    selected_websites = [user_websites[i] for i in selected_website_indices]
                    
                    # Perform async scraping
                    async with aiohttp.ClientSession() as session:
                        scraped_articles = await scrape_selected_websites(session, selected_websites)
                    
                    # Display results
                    st.subheader(f"Total websites Scraped: {len(scraped_articles)}")
                    if scraped_articles:
                        results_df = pd.DataFrame(scraped_articles)
                        st.dataframe(results_df)
                else:
                    st.warning("Please select websites to scrape")
        
        with col2:
            # Scrape All Websites Button
            if st.button("Scrape All"):
                if user_websites:
                    # Perform async scraping on all websites
                    async with aiohttp.ClientSession() as session:
                        scraped_articles = await scrape_selected_websites(session, user_websites)
                    
                    # Display results
                    st.subheader(f"Total websites Scraped: {len(scraped_articles)}")
                    if scraped_articles:
                        results_df = pd.DataFrame(scraped_articles)
                        st.dataframe(results_df)
                else:
                    st.warning("No websites available to scrape")

def main():
    asyncio.run(render_crawl_page())

if __name__ == "__main__":
    main()