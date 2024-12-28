import streamlit as st
import requests
from pathlib import Path
import json
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def fetch_user_scrapping_prompt():
    """Fetch scrapping prompt for the current user"""
    try:
        # Ensure user is logged in
        if not st.session_state.get("username_id"):
            st.warning("Please log in to view scrapping prompts")
            return None
        
        # Prepare the API call to fetch scrapping prompt
        api_url = f"{st.secrets.get('API_BASE_URL', 'http://localhost:8000')}/db/query/scrapping_prompts"
        
        # Conditions to filter by user_id
        conditions = {"user_id": st.session_state.username_id}
        
        # Columns to fetch
        columns = [
            'articles_links_prompt', 'get_login_url_prompt', 
            'classification_extraction_prompt', 'login_selectors_prompt', 
            'select_relevant_link', 'is_article_relevant'
        ]
        
        # Make the API request
        response = requests.post(
            api_url, 
            json={
                "conditions": conditions,
                "columns": columns
            }
        )
        
        if response.status_code == 200:
            results = response.json()
            return results[0] if results else None
        else:
            st.error(f"Failed to fetch scrapping prompts: {response.text}")
            return None
    
    except Exception as e:
        st.error(f"Error fetching scrapping prompts: {e}")
        return None

def update_scrapping_prompt():
    """Update or create scrapping prompt for the user"""
    st.subheader("Scrapping Prompts Configuration")
    
    # Fetch existing scrapping prompt
    existing_prompt = fetch_user_scrapping_prompt()
    
    # Prepare update payload
    updated_prompt = {}
    
    # Function to calculate height based on text content
    def calculate_height(text):
        # Base height + additional height based on number of lines
        lines = text.count('\n') + 1 if text else 1
        return max(100, min(400, lines * 25))  # Min 100, Max 400, 25px per line
    
    # Input fields for each scrapping prompt component
    updated_prompt['articles_links_prompt'] = st.text_area(
        "Articles Links Prompt", 
        value=existing_prompt.get('articles_links_prompt', '') if existing_prompt else '',
        height=calculate_height(existing_prompt.get('articles_links_prompt', '') if existing_prompt else '')
    )
    
    updated_prompt['get_login_url_prompt'] = st.text_area(
        "Get Login URL Prompt", 
        value=existing_prompt.get('get_login_url_prompt', '') if existing_prompt else '',
        height=calculate_height(existing_prompt.get('get_login_url_prompt', '') if existing_prompt else '')
    )
    
    updated_prompt['classification_extraction_prompt'] = st.text_area(
        "Classification Extraction Prompt", 
        value=existing_prompt.get('classification_extraction_prompt', '') if existing_prompt else '',
        height=calculate_height(existing_prompt.get('classification_extraction_prompt', '') if existing_prompt else '')
    )
    
    updated_prompt['login_selectors_prompt'] = st.text_area(
        "Login Selectors Prompt", 
        value=existing_prompt.get('login_selectors_prompt', '') if existing_prompt else '',
        height=calculate_height(existing_prompt.get('login_selectors_prompt', '') if existing_prompt else '')
    )
    
    updated_prompt['select_relevant_link'] = st.text_area(
        "Select Relevant Link Prompt", 
        value=existing_prompt.get('select_relevant_link', '') if existing_prompt else '',
        height=calculate_height(existing_prompt.get('select_relevant_link', '') if existing_prompt else '')
    )
    
    updated_prompt['is_article_relevant'] = st.text_area(
        "Is Article Relevant Prompt", 
        value=existing_prompt.get('is_article_relevant', '') if existing_prompt else '',
        height=calculate_height(existing_prompt.get('is_article_relevant', '') if existing_prompt else '')
    )
    
    # Add user_id to the prompt
    updated_prompt['user_id'] = st.session_state.username_id
    
    if st.button("Save Scrapping Prompts"):
        try:
            # Determine whether to use update or add endpoint
            if existing_prompt:
                # Update existing prompt
                api_url = f"{st.secrets.get('API_BASE_URL', 'http://localhost:8000')}/db/update_rows/scrapping_prompts"
                payload = {
                    "conditions": {
                        "user_id": st.session_state.username_id
                    },
                    "updated_values": updated_prompt
                }
            else:
                # Add new prompt
                api_url = f"{st.secrets.get('API_BASE_URL', 'http://localhost:8000')}/db/add_rows/scrapping_prompts"
                payload = [updated_prompt]
            
            # Make the API request
            response = requests.post(api_url, json=payload)
            
            if response.status_code == 200:
                st.success("Scrapping prompts saved successfully!")
                st.rerun()
            else:
                st.error(f"Failed to save scrapping prompts: {response.text}")
        
        except Exception as e:
            st.error(f"Error saving scrapping prompts: {e}")

def render_scrapping_prompts_page():
    st.title("Scrapping Prompts Manager")

    # Ensure user is logged in
    if not st.session_state.get("username_id"):
        st.warning("Please log in to manage scrapping prompts")
        return

    # Add custom CSS (if needed)
    st.markdown("""
        <style>
        /* Add any custom styles here */
        </style>
    """, unsafe_allow_html=True)

    # Current scrapping prompt view
    st.subheader("Current Scrapping Prompts")
    current_prompt = fetch_user_scrapping_prompt()
    
    if current_prompt:
        # Display current prompts in an expander
        with st.expander("View Current Scrapping Prompts"):
            st.json(current_prompt)
    else:
        st.info("No existing scrapping prompts. Create a new configuration.")

    # Update/Create Scrapping Prompts
    update_scrapping_prompt()

if __name__ == "__main__":
    render_scrapping_prompts_page()
