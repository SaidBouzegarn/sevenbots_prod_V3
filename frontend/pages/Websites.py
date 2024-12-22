import streamlit as st
import requests
from pathlib import Path
import json
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def fetch_user_websites():
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
        
        # Make the API request
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

def add_new_website():
    """Add a new website for the user"""
    st.subheader("Add New Website")
    
    # Use a form to batch elements and manage state
    with st.form(key='add_website_form'):
        # Website URL and Login Details
        url = st.text_input("Website URL")
        login_url = st.text_input("Login URL")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        # Selectors for Login Process
        st.subheader("Login Selectors")
        username_selector = st.text_input("Username Selector (CSS/XPath)")
        password_selector = st.text_input("Password Selector (CSS/XPath)")
        submit_button_selector = st.text_input("Submit Button Selector (CSS/XPath)")
        
        # Validation Checkboxes
        st.subheader("Validation")
        login_url_ok = st.checkbox("Login URL Verified")
        username_selector_ok = st.checkbox("Username Selector Verified")
        password_selector_ok = st.checkbox("Password Selector Verified")
        submit_button_selector_ok = st.checkbox("Submit Button Selector Verified")
        
        # Additional Status
        logged_in = st.checkbox("Currently Logged In")
        
        # Form submit button
        submitted = st.form_submit_button("Add Website")
        
        if submitted:
            # Validate required fields
            if not url:
                st.error("Website URL is required")
                return
            
            # Prepare website data for API
            new_website = {
                'url': url,
                'login_url': login_url,
                'username': username,
                'password': password,
                'username_selector': username_selector,
                'password_selector': password_selector,
                'submit_button_selector': submit_button_selector,
                'login_url_ok': login_url_ok,
                'username_selector_ok': username_selector_ok,
                'password_selector_ok': password_selector_ok,
                'submit_button_selector_ok': submit_button_selector_ok,
                'logged_in': logged_in,
                'user_id': st.session_state.username_id
            }
            
            try:
                api_url = f"{st.secrets.get('API_BASE_URL', 'http://localhost:8000')}/db/add_rows/websites"
                
                response = requests.post(
                    api_url, 
                    json=[new_website]  # Wrap in a list as the API expects a list of rows
                )
                
                if response.status_code == 200:
                    st.success("Website added successfully!")
                    st.rerun()  # Refresh the page
                else:
                    st.error(f"Failed to add website: {response.text}")
            
            except Exception as e:
                st.error(f"Error adding website: {e}")

def edit_website():
    """Edit an existing website"""
    st.subheader("Edit Website")
    
    # Fetch user's websites
    websites = fetch_user_websites()
    
    if not websites:
        st.warning("No websites to edit")
        return
    
    # Select website to edit
    selected_website_index = st.selectbox(
        "Select Website to Edit", 
        range(len(websites)), 
        format_func=lambda x: f"{websites[x].get('url', 'Unnamed Website')}"
    )
    
    selected_website = websites[selected_website_index]
    
    # Use a form to batch elements and manage state
    with st.form(key='edit_website_form'):
        # Website URL and Login Details
        url = st.text_input("Website URL", value=selected_website.get('url', ''))
        login_url = st.text_input("Login URL", value=selected_website.get('login_url', ''))
        username = st.text_input("Username", value=selected_website.get('username', ''))
        password = st.text_input("Password", value=selected_website.get('password', ''), type="password")
        
        # Selectors for Login Process
        st.subheader("Login Selectors")
        username_selector = st.text_input("Username Selector (CSS/XPath)", value=selected_website.get('username_selector', ''))
        password_selector = st.text_input("Password Selector (CSS/XPath)", value=selected_website.get('password_selector', ''))
        submit_button_selector = st.text_input("Submit Button Selector (CSS/XPath)", value=selected_website.get('submit_button_selector', ''))
        
        # Validation Checkboxes
        st.subheader("Validation")
        login_url_ok = st.checkbox("Login URL Verified", value=selected_website.get('login_url_ok', False))
        username_selector_ok = st.checkbox("Username Selector Verified", value=selected_website.get('username_selector_ok', False))
        password_selector_ok = st.checkbox("Password Selector Verified", value=selected_website.get('password_selector_ok', False))
        submit_button_selector_ok = st.checkbox("Submit Button Selector Verified", value=selected_website.get('submit_button_selector_ok', False))
        
        # Additional Status
        logged_in = st.checkbox("Currently Logged In", value=selected_website.get('logged_in', False))
        
        # Form submit button
        submitted = st.form_submit_button("Update Website")
        
        if submitted:
            try:
                api_url = f"{st.secrets.get('API_BASE_URL', 'http://localhost:8000')}/db/update_rows/websites"
                
                # Prepare update payload
                update_payload = {
                    "conditions": {
                        "url": selected_website.get('url'),
                        "user_id": st.session_state.username_id
                    },
                    "updated_values": {
                        'url': url,
                        'login_url': login_url,
                        'username': username,
                        'password': password,
                        'username_selector': username_selector,
                        'password_selector': password_selector,
                        'submit_button_selector': submit_button_selector,
                        'login_url_ok': login_url_ok,
                        'username_selector_ok': username_selector_ok,
                        'password_selector_ok': password_selector_ok,
                        'submit_button_selector_ok': submit_button_selector_ok,
                        'logged_in': logged_in
                    }
                }
                
                response = requests.post(api_url, json=update_payload)
                
                if response.status_code == 200:
                    st.success("Website updated successfully!")
                    st.rerun()  # Refresh the page
                else:
                    st.error(f"Failed to update website: {response.text}")
            
            except Exception as e:
                st.error(f"Error updating website: {e}")

def delete_website():
    """Delete an existing website"""
    st.subheader("Delete Website")
    
    # Fetch user's websites
    websites = fetch_user_websites()
    
    if not websites:
        st.warning("No websites to delete")
        return
    
    # Select website to delete
    selected_website_index = st.selectbox(
        "Select Website to Delete", 
        range(len(websites)), 
        format_func=lambda x: f"{websites[x].get('url', 'Unnamed Website')}",
    )
    
    selected_website = websites[selected_website_index]
    
    # Confirmation before deletion
    if st.button("Confirm Delete"):
        try:
            api_url = f"{st.secrets.get('API_BASE_URL', 'http://localhost:8000')}/db/delete_rows/websites"
            
            # Prepare delete conditions
            delete_conditions = {
                "url": selected_website.get('url'),
                "user_id": st.session_state.username_id
            }
            
            response = requests.post(api_url, json=delete_conditions)
            
            if response.status_code == 200:
                st.success("Website deleted successfully!")
                st.rerun()  # Refresh the page
            else:
                st.error(f"Failed to delete website: {response.text}")
        
        except Exception as e:
            st.error(f"Error deleting website: {e}")

def render_websites_page():
    st.title("Website Login Manager")

    # Ensure user is logged in
    if not st.session_state.get("username_id"):
        st.warning("Please log in to manage websites")
        return

    # Create tabs for different website management actions
    tab1, tab2, tab3, tab4 = st.tabs([
        "View Websites", 
        "Add New Website", 
        "Edit Website", 
        "Delete Website"
    ])

    with tab1:
        st.subheader("Your Websites")
        websites = fetch_user_websites()
        
        if websites:
            # Display websites in a more readable format
            for website in websites:
                with st.expander(f"Website: {website.get('url', 'Unnamed')}"):
                    # Mask sensitive information
                    display_website = website.copy()
                    display_website['password'] = '********' if display_website.get('password') else ''
                    st.json(display_website)
        else:
            st.info("No websites found. Add a new website to get started!")

    with tab2:
        add_new_website()

    with tab3:
        edit_website()

    with tab4:
        delete_website()

if __name__ == "__main__":
    render_websites_page()
