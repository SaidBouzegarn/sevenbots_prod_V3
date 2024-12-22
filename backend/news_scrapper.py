import os
import sys
import asyncio
import requests
from urllib.parse import urljoin
import httpx

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from playwright.async_api import async_playwright
from playwright_stealth import stealth_async
import os
from backend.utils.utils import extract_content, clean_html_for_login_detection
from backend.utils.llm_utils import select_likely_URLS, detect_login_url, detect_selectors, classify_and_extract_news_article
from datetime import datetime
from collections import deque
from urllib.parse import urlparse
import traceback
import dotenv
import time
import random
from pybloom_live import ScalableBloomFilter
import logging
import asyncio
import pandas as pd
from jinja2 import Template
import aiohttp
import logging
import json
from backend.db_manager import AsyncRDSPostgresManager
logger = logging.getLogger(__name__)

dotenv.load_dotenv()
# Base URL for the API
BASE_URL = "http://localhost:8000"  # Adjust this to match your FastAPI server's address

class NewsScrapper:
    def __init__(self, user_id, website_url, login_url=None, username=None, password=None,
                 username_selector=None, password_selector=None, submit_button_selector=None, login_url_ok=False,
                username_selector_ok=False, password_selector_ok=False, submit_button_selector_ok=False, logged_in = False, crawl=True, max_pages=25):
        self.user_id = user_id
        self.website_url = website_url
        self.domain = self._extract_domain(self.website_url)
        self.login_url = login_url
        self.username = username
        self.password = password
        self.username_selector = username_selector
        self.password_selector = password_selector
        self.submit_button_selector = submit_button_selector
        self.logged_in = logged_in
        self.username_selector_ok = username_selector_ok
        self.password_selector_ok = password_selector_ok
        self.submit_button_selector_ok = submit_button_selector_ok
        self.login_url_ok = login_url_ok
        self.crawl = crawl
        self.max_pages = max_pages
        self.session_cookies = None

        self.headless = True
        # Async-specific attributes
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        
        # Initialize Bloom filter
        self.visited_urls = ScalableBloomFilter(
            mode=ScalableBloomFilter.LARGE_SET_GROWTH,
            error_rate=0.01
        )
        
        # Store the list of visited URLs
        self.visited_urls_list = []
        self.prompts = None

        # Add PostgresManager initialization
        self.postgresmanager = AsyncRDSPostgresManager()

    @classmethod
    async def create(cls, **kwargs):
        """
        Async class method to create an instance of NewsScrapper
        """
        # Create the instance
        instance = cls(**kwargs)
        # Pre-populate from database
        instance.visited_urls_list = await instance.get_visited_urls()
        instance.prompts = await instance.get_prompts()
        # Add each URL individually to the Bloom filter            
        for url in instance.visited_urls_list:
            if not url in instance.visited_urls:
                instance.visited_urls.add(url)

        return instance

    async def initialize(self):
        """
        Async initialization method to set up Playwright browser
        """
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--disable-extensions",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-setuid-sandbox",
            ],
        )

        # Create a new context with stealth settings
        self.context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) " +
                       "AppleWebKit/537.36 (KHTML, like Gecko) " +
                       "Chrome/96.0.4664.45 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
        )

        # Open a new page and navigate to the website
        self.page = await self.context.new_page()
        await self.page.goto(self.website_url)
        await self.page.wait_for_load_state('networkidle')
        
        # Apply stealth techniques
        await stealth_async(self.page)

        # Check authentication requirements and login if needed
        if await self._can_authenticate():
            if await self._initialize_login():
                await self.login()
        else:
            logger.info("No authentication credentials provided. Running in anonymous mode.")

    async def login(self):
        # Navigate to the login page
        await self.page.goto(self.login_url)

        # Wait for the login page to load
        await self.page.wait_for_load_state('networkidle')

        # Fill in the username and password fields
        await self.page.fill(self.username_selector, self.username)
        await self.page.fill(self.password_selector, self.password)

        # Click the submit button
        await self.page.click(self.submit_button_selector)

        # Wait for the navigation to complete
        await self.page.wait_for_load_state('networkidle')

        # Add session verification
        await self._verify_session()

    async def _verify_session(self):
        """Verify that the session is still active."""
        # Store cookies after successful login
        self.session_cookies = await self.context.cookies()
        # Basic session check
        if not self.session_cookies:
            raise Exception("No session cookies found after login")
        else: 
            self.logged_in = True

    async def scrape(self):
        # Only restore cookies if they exist and are valid
        if hasattr(self, 'session_cookies') and self.session_cookies:
            await self.context.add_cookies(self.session_cookies)
        
        responses = []
        await asyncio.sleep(random.uniform(1, 10))
        await self.page.goto(self.website_url)
        await self.page.wait_for_load_state('networkidle')  # Wait for the page to be fully loaded
        
        urls = await extract_content(self.page, output_type="links")
        logger.info(f"found links {(urls)} ")
        
        # Check if the specific prompt exists
        articles_links_prompt_template = self.prompts.get('articles_links_prompt')
        # Create Jinja2 template from the prompt text
        template = Template(articles_links_prompt_template)
        
        # Render the template with the URLs
        prompt = template.render(
            urls=urls
        )
        
        # Use the rendered prompt with your existing function
        response = await select_likely_URLS(prompt)

        lucky_urls = response.likely_urls
        url_list = [link['href'] for link in urls]

        # Filter lucky_urls to only include URLs that exist in 'urls'
        if isinstance(lucky_urls, list):
            n_lucky_urls = [url for url in lucky_urls if url in url_list]
        else:
            n_lucky_urls = [lucky_urls] if lucky_urls in url_list else []

        logger.info(f"found {len(n_lucky_urls)} new urls")
        logger.info(f"llm hallucinated {len(lucky_urls) - len(n_lucky_urls)} urls")
        
        # Initialize deque for efficient popping from the left
        to_visit = deque()
        to_visit.extend(n_lucky_urls)

        # Navigate to the target URL while staying logged in
        await self.page.wait_for_load_state('networkidle')  # Wait for the page to be fully loaded
        structured_content = await extract_content(self.page, output_type="formatted_text")
        
        # Check if the specific prompt exists
        classification_prompt_template = self.prompts.get('classification_extraction_prompt')
        
        # Create Jinja2 template from the prompt text
        template = Template(classification_prompt_template)
        
        # Render the template with the structured content
        prompt = template.render(
            cleaned_html=structured_content
        )
        
        # Use the rendered prompt with your existing function
        response = await classify_and_extract_news_article(prompt)
        responses.append((self.website_url, response.model_dump()))
        
        if not self.crawl: 
            await self.add_visited_urls([(self.website_url, response.classification if response else None)])
            return await self.add_responses(llm_function="classify_and_extract_news_article", responses=responses)
        
        newly_visited_urls = [(self.website_url, response.classification if response else None)]

        while to_visit and len(responses) < self.max_pages:
            current_url = to_visit.popleft()
            logger.info(f"crawling url {current_url}")
            is_visited_url = await self.is_url_visited(current_url)
            if is_visited_url:
                continue

            # Add delay before each new page navigation
            await asyncio.sleep(random.uniform(1, 10))
            await self.page.goto(current_url)

            try:
                await self.page.wait_for_load_state('networkidle')  # Wait for the page to be fully loaded

                html_content = await extract_content(self.page, output_type="formatted_text")
                classification_prompt_template = self.prompts.get('classification_extraction_prompt')
                
                # Create Jinja2 template from the prompt text
                template = Template(classification_prompt_template)
                
                # Render the template with the structured content
                prompt = template.render(
                    cleaned_html=structured_content
                )
                response = await classify_and_extract_news_article(prompt)
                if response: 
                    responses.append((current_url, response.model_dump()))
                newly_visited_urls.append((current_url, response.classification if response else False))
            except Exception as e: 
                logger.info(f"Warning: Could not classify and extract news article from {current_url}. Skipping... Full error:\n{traceback.format_exc()}")
            
            await self.page.wait_for_load_state('networkidle')  # Wait for the page to be fully loaded

            new_links = await extract_content(self.page, output_type="links")
            logger.info(f"found {len(new_links)} new links")
            articles_links_prompt_template = self.prompts.get('articles_links_prompt')
            template = Template(str(articles_links_prompt_template))
            prompt = template.render(
                urls=new_links
            )
            try: 
                response = await select_likely_URLS(prompt)
                new_lucky_urls = response.likely_urls

                # Filter new_lucky_urls to only include URLs that exist in 'new_links'
                if isinstance(new_lucky_urls, list):
                    n_lucky_urls = [url for url in new_lucky_urls if url in new_links]
                else:
                    n_lucky_urls = [new_lucky_urls] if new_lucky_urls in new_links else []

                logger.info(f"found {len(n_lucky_urls)} new urls")
                logger.info(f"llm hallucinated {len(new_lucky_urls) - len(n_lucky_urls)} urls")

                to_visit.extend(n_lucky_urls)
            except Exception as e: 
                logger.info(f"Warning: Could not select likely URLs from {current_url}. Full error:\n{traceback.format_exc()}")

        r = await self.add_visited_urls(newly_visited_urls)
        print("saving visited urls", r)
        r= await self.add_responses(llm_function="classify_and_extract_news_article", responses=responses)
        return ( len(newly_visited_urls), len([response for response in responses if response[1].get('classification') == True]), len([response for response in responses if response[1].get('classification') == False]))
    async def close(self):
        """Async method to close browser and playwright"""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def _can_authenticate(self):
        """Check if authentication is possible with provided credentials."""
        return self.username is not None and self.password is not None

    async def _initialize_login(self):
        """Initialize login requirements before attempting authentication."""
        if not self.login_url_ok:
            try: 
                response = await self.get_login_url()
                if response.sucess:
                    self.login_url = response.login_url
                    self.login_url_ok = True
                    self.add_responses(llm_function="get_login_url", responses=(self.website_url, response.model_dump()))

                else:
                    logger.info(f"Warning: Could not determine login URL. Continuing without authentication...Full error:\n{traceback.format_exc()}")
                    return False
            except Exception as e:
                logger.info(f"Warning: Could not determine login URL. Continuing without authentication...Full error:\n{traceback.format_exc()}")
                return False
        if not self.login_url_ok:
            logger.info(f"Warning: Login URL is not a string and its type is {type(self.login_url)}. Continuing without authentication...")
            return False
        
        # Verify all required selectors are present
        if not self.username_selector_ok or not self.password_selector_ok or not self.submit_button_selector_ok:
            try:
                # Detect login selectors
                response = await self.get_login_selectors()
                if not response.sucess:  # Check if any selector is None or empty
                    logger.info("Warning: One or more login selectors are empty or invalid")
                    return False
                self.add_responses(llm_function="get_login_selectors", responses=(self.login_url, response.model_dump()))
                self.username_selector = response.username_selector
                self.password_selector = response.password_selector
                self.submit_button_selector = response.submit_button_selector
                self.username_selector_ok = True
                self.password_selector_ok = True
                self.submit_button_selector_ok = True
                return True
                
            except Exception as e:
                logger.info(f"Error detecting login selectors: {e}")
                return False

        return True  # Add explicit return True when all checks pass

    async def get_login_url(self):
        """Find and return the login URL for the website."""
        links = await extract_content(self.page, output_type="links")
        html_content = await extract_content(self.page, output_type="full_html")

        prompt_template = self.prompts.get('get_login_url_prompt')
        template = Template(prompt_template)
        prompt = template.render(
            links = links
        )

        response = await detect_login_url(prompt)
        return response.login_url
    
    async def get_login_selectors(self):
        await self.page.goto(self.login_url)
        await self.page.wait_for_load_state('networkidle')
        
        # Take and save screenshot
        screenshots_dir = os.path.join(os.path.dirname(__file__), '..', 'Data', 'screenshots')
        os.makedirs(screenshots_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        screenshot_path = os.path.join(screenshots_dir, f'login_page_{self.domain}_{timestamp}.png')
        await self.page.screenshot(path=screenshot_path)
        logger.info(f"Screenshot saved to: {screenshot_path}")
        
        # Extract page content
        page_content = await extract_content(self.page, output_type="full_html")
        # Clean HTML for login detection
        cleaned_html = clean_html_for_login_detection(page_content)
        # Generate prompt for login selectors

        prompt_template = self.prompts.get('login_selectors_prompt')
        template = Template(prompt_template)
        prompt = template.render(
            html=cleaned_html
        )
        # Detect login selectors
        response = await detect_selectors(prompt)
        return response

    async def get_visited_urls_api(self):
        """
        Retrieve all visited URLs for the current domain and user using the databases API service
        Returns:
            list: List of visited URLs
        """
        try:
            # Prepare query conditions
            conditions = {
                "domain": self.domain,
                "user_id": self.user_id  
            }

            # Use the new query_table endpoint
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BASE_URL}/db/query/visited_urls", 
                    json={
                        "conditions": conditions,
                        "columns": ["url"]  # Only retrieve URLs
                    }
                )

                if response.status_code != 200:
                    logger.info(f"Error querying visited URLs: {response.text}")
                    return set()

                results = response.json()
                
                # Extract URLs from the results
                urls = set(row['url'] for row in results)
                
                return urls
        
        except Exception as e:
            logger.info(f"Error retrieving visited URLs via API: {e}")
            return set()

    async def add_visited_urls_api(self, urls):
        """
        Add visited URLs to the database via API service
        
        Args:
            urls (list): List of tuples containing (url, classification)
        """
        try:
            # Prepare the data for insertion
            data = [
                {
                    "user_id": self.user_id,
                    "domain": self.domain,
                    "url": str(url),
                    "is_article": classification,
                } 
                for url, classification in urls
            ]

            # Use the new add_rows endpoint
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BASE_URL}/db/add_rows/visited_urls", 
                    content=json.dumps(data),
                    headers={"Content-Type": "application/json"}
                )

                # Check if the request was successful
                if response.status_code != 200:
                    logger.info(f"Error adding visited URLs: {response.text}")
                    return False

                return True
        
        except Exception as e:
            logger.info(f"Error adding visited URLs via API: {e}")
            return False

    async def is_url_visited(self, url):
        # Probabilistic check
        return url in self.visited_urls


    def _extract_domain(self, url):
        """
        Extract the domain from a given URL.
        
        Args:
            url (str): The full URL to extract domain from
        
        Returns:
            str: The extracted domain name
        """
        try:
            from urllib.parse import urlparse
            
            # Parse the URL
            parsed_url = urlparse(url)
            
            # Extract the domain
            domain = parsed_url.netloc
            
            # Remove 'www.' if present
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain
        
        except Exception as e:
            logger.info(f"Error extracting domain from {url}: {e}")
            return url  # Fallback to returning the original URL if parsing fails
    
    async def get_prompts_api(self):
        """
        Retrieve prompts from database for the specific domain
        
        Returns:
            dict: A dictionary containing the most recent prompts
        """
        try:
            # Prepare query conditions
            conditions = {
                "user_id": self.user_id,
            }

            # Use the new query_table endpoint
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BASE_URL}/db/query/scrapping_prompts", 
                    json={
                        "conditions": conditions,
                        "order_by": [{"column": "time_stamp", "direction": "DESC"}],
                        "limit": 1  # Only get the most recent row
                    }
                )
            
            # Check if the request was successful
            if response.status_code != 200:
                logger.info(f"Error querying prompts: {response.text}")
                return {}

            # Parse the response
            results = response.json()
            
            # If no results, return empty dictionary
            if not results:
                logger.info(f"No prompts found for user {self.user_id}")
                return {}

            # Get the most recent prompts row
            latest_prompts = results[0]
            
            # Return a dictionary with prompt templates
            return {
                'articles_links_prompt': latest_prompts.get('articles_links_prompt', ''),
                'classification_extraction_prompt': latest_prompts.get('classification_extraction_prompt', ''),
                'get_login_url_prompt': latest_prompts.get('get_login_url_prompt', ''),
                'login_selectors_prompt': latest_prompts.get('login_selectors_prompt', '')
            }
            
        except Exception as e:
            logger.info(f"Error retrieving prompts: {e}")
            raise e

    async def add_responses_api(self, llm_function, responses):
        """
        Add LLM call responses to the database via API service
        
        Args:
            llm_function (str): Name of the LLM function used
            responses (list): List of tuples containing (prompt, response)
        """
        try:
            # Prepare the data for insertion
            data = [
                {
                    "user_id": self.user_id,
                    "llm": llm_function,
                    "prompt": str(prompt),
                    "response": str(response),
                    "time": str(datetime.now().isoformat())
                } 
                for prompt, response in responses
            ]

            # Use the new add_rows endpoint
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BASE_URL}/db/add_rows/llm_calls", 
                    content=json.dumps(data),
                    headers={"Content-Type": "application/json"}
                )
            
            # Check if the request was successful
            if response.status_code != 200:
                logger.info(f"Error adding LLM responses: {response.text}")
                return False

            return True
        
        except Exception as e:
            logger.info(f"Error adding LLM responses via API: {e}")
            return False

    async def get_visited_urls(self):
        """
        Retrieve all visited URLs directly from the database for the current domain and user
        """
        try:
            conditions = {
                "domain": self.domain,
                "user_id": self.user_id
            }
            results = await self.postgresmanager.query_table(
                "visited_urls",
                conditions=conditions,
                columns=["url"]
            )
            return set(row['url'] for row in results)
        except Exception as e:
            logger.info(f"Error retrieving visited URLs directly from database: {e}")
            return set()

    async def get_prompts(self):
        """
        Retrieve prompts directly from database for the specific domain
        """
        try:
            conditions = {
                "user_id": self.user_id
            }
            results = await self.postgresmanager.query_table(
                "scrapping_prompts",
                conditions=conditions,
                columns=[
                    'articles_links_prompt',
                    'classification_extraction_prompt',
                    'get_login_url_prompt',
                    'login_selectors_prompt'
                ]
            )
            
            if not results:
                logger.info(f"No prompts found for user {self.user_id}")
                return {}

            # Return the most recent prompts
            return {
                'articles_links_prompt': results[0].get('articles_links_prompt', ''),
                'classification_extraction_prompt': results[0].get('classification_extraction_prompt', ''),
                'get_login_url_prompt': results[0].get('get_login_url_prompt', ''),
                'login_selectors_prompt': results[0].get('login_selectors_prompt', '')
            }
        except Exception as e:
            logger.info(f"Error retrieving prompts directly from database: {e}")
            raise e

    async def add_visited_urls(self, urls):
        """
        Add visited URLs directly to the database
        
        Args:
            urls (list): List of tuples containing (url, classification)
        """
        try:
            data = [
                {
                    "user_id": self.user_id,
                    "domain": self.domain,
                    "url": str(url),
                    "is_article": classification,
                } 
                for url, classification in urls
            ]
            
            await self.postgresmanager.add_rows("visited_urls", data)
            return True
        except Exception as e:
            logger.info(f"Error adding visited URLs directly to database: {e}")
            return False

    async def add_responses(self, llm_function, responses):
        """
        Add LLM call responses directly to the database
        
        Args:
            llm_function (str): Name of the LLM function used
            responses (list): List of tuples containing (prompt, response)
        """
        try:
            data = [
                {
                    "user_id": self.user_id,
                    "llm": llm_function,
                    "prompt": str(prompt),
                    "response": str(response),
                    "time": str(datetime.now().isoformat())
                } 
                for prompt, response in responses
            ]
            
            await self.postgresmanager.add_rows("llm_calls", data)
            return True
        except Exception as e:
            logger.info(f"Error adding LLM responses directly to database: {e}")
            return False


