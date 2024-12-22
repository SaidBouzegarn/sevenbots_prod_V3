import streamlit as st
import requests
from pathlib import Path
import json
import logging
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add a new function to fetch user's prompts
# Add a new function to fetch user's prompts
def fetch_user_prompts():
    """Fetch prompts for the current user"""
    try:
        # Ensure user is logged in
        if not st.session_state.get("username_id"):
            st.warning("Please log in to view prompts")
            return []
        
        # Prepare the API call to fetch prompts
        api_url = f"{st.secrets.get('API_BASE_URL', 'http://localhost:8000')}/db/query/prompts"
        
        # Conditions to filter by user_id
        conditions = {"user_id": st.session_state.username_id}
        
        # Columns to exclude user_id
        columns = [
            'id', 'agent_name', 'agent_level', 'assistant_prompt', 
            'decision_prompt', 'system_prompt', 'agent_llm', 
            'agent_max_tokens', 'agent_temperature', 'assistant_llm', 
            'assistant_max_tokens', 'assistant_temperature', 
            'agent_supervisor', 'agent_subordinates'
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
            st.error(f"Failed to fetch prompts: {response.text}")
            return []
    
    except Exception as e:
        st.error(f"Error fetching prompts: {e}")
        return []

def add_new_prompt():
    """Add a new prompt for the user"""
    st.subheader("Add New Prompt")
    
    # Create input fields for each prompt attribute
    new_prompt = {}
    
    # Basic fields
    new_prompt['agent_name'] = st.text_input("Agent Name")
    new_prompt['agent_level'] = st.text_input("Agent Level")
    
    # Prompt fields
    new_prompt['assistant_prompt'] = st.text_area("Assistant Prompt")
    new_prompt['decision_prompt'] = st.text_area("Decision Prompt")
    new_prompt['system_prompt'] = st.text_area("System Prompt")
    
    # LLM Configuration
    new_prompt['agent_llm'] = st.text_input("Agent LLM")
    new_prompt['agent_max_tokens'] = st.text_input("Agent Max Tokens")
    new_prompt['agent_temperature'] = st.text_input("Agent Temperature")
    
    # Additional fields
    new_prompt['assistant_llm'] = st.text_input("Assistant LLM")
    new_prompt['assistant_max_tokens'] = st.text_input("Assistant Max Tokens")
    new_prompt['assistant_temperature'] = st.text_input("Assistant Temperature")
    
    new_prompt['agent_supervisor'] = st.text_input("Agent Supervisor")
    new_prompt['agent_subordinates'] = st.text_input("Agent Subordinates")
    
    # Add user_id to the prompt
    new_prompt['user_id'] = st.session_state.username_id
    
    if st.button("Add Prompt"):
        try:
            api_url = f"{st.secrets.get('API_BASE_URL', 'http://localhost:8000')}/db/add_rows/prompts"
            
            response = requests.post(
                api_url, 
                json=[new_prompt]  # Wrap in a list as the API expects a list of rows
            )
            
            if response.status_code == 200:
                st.success("Prompt added successfully!")
                st.rerun()  # Refresh the page
            else:
                st.error(f"Failed to add prompt: {response.text}")
        
        except Exception as e:
            st.error(f"Error adding prompt: {e}")

def edit_prompt():
    """Edit an existing prompt"""
    st.subheader("Edit Prompt")
    
    # Fetch user's prompts
    prompts = fetch_user_prompts()
    
    if not prompts:
        st.warning("No prompts to edit")
        return
    
    # Select prompt to edit
    selected_prompt_index = st.selectbox(
        "Select Prompt to Edit", 
        range(len(prompts)), 
        format_func=lambda x: f"{prompts[x].get('agent_name', 'Unnamed')}"
    )
    
    selected_prompt = prompts[selected_prompt_index]
    
    # Create editable fields
    edited_prompt = {}
    
    # Editable fields (similar to add_new_prompt)
    edited_prompt['agent_name'] = st.text_input("Agent Name", value=selected_prompt.get('agent_name', ''))
    edited_prompt['agent_level'] = st.text_input("Agent Level", value=selected_prompt.get('agent_level', ''))
    
    edited_prompt['assistant_prompt'] = st.text_area("Assistant Prompt", value=selected_prompt.get('assistant_prompt', ''))
    edited_prompt['decision_prompt'] = st.text_area("Decision Prompt", value=selected_prompt.get('decision_prompt', ''))
    edited_prompt['system_prompt'] = st.text_area("System Prompt", value=selected_prompt.get('system_prompt', ''))
    
    edited_prompt['agent_llm'] = st.text_input("Agent LLM", value=selected_prompt.get('agent_llm', ''))
    edited_prompt['agent_max_tokens'] = st.text_input("Agent Max Tokens", value=selected_prompt.get('agent_max_tokens', ''))
    edited_prompt['agent_temperature'] = st.text_input("Agent Temperature", value=selected_prompt.get('agent_temperature', ''))
    
    edited_prompt['assistant_llm'] = st.text_input("Assistant LLM", value=selected_prompt.get('assistant_llm', ''))
    edited_prompt['assistant_max_tokens'] = st.text_input("Assistant Max Tokens", value=selected_prompt.get('assistant_max_tokens', ''))
    edited_prompt['assistant_temperature'] = st.text_input("Assistant Temperature", value=selected_prompt.get('assistant_temperature', ''))
    
    edited_prompt['agent_supervisor'] = st.text_input("Agent Supervisor", value=selected_prompt.get('agent_supervisor', ''))
    edited_prompt['agent_subordinates'] = st.text_input("Agent Subordinates", value=selected_prompt.get('agent_subordinates', ''))
    
    if st.button("Update Prompt"):
        try:
            api_url = f"{st.secrets.get('API_BASE_URL', 'http://localhost:8000')}/db/update_rows/prompts"
            
            # Prepare update payload
            update_payload = {
                "conditions": {
                    "id": selected_prompt.get('id'),
                    "user_id": st.session_state.username_id
                },
                "updated_values": edited_prompt
            }
            
            response = requests.post(api_url, json=update_payload)
            
            if response.status_code == 200:
                st.success("Prompt updated successfully!")
                st.rerun()  # Replace st.experimental_rerun() with st.rerun()
            else:
                st.error(f"Failed to update prompt: {response.text}")
        
        except Exception as e:
            st.error(f"Error updating prompt: {e}")

def delete_prompt():
    """Delete an existing prompt"""
    st.subheader("Delete Prompt")
    
    # Fetch user's prompts
    prompts = fetch_user_prompts()
    
    if not prompts:
        st.warning("No prompts to delete")
        return
    
    # Select prompt to delete
    selected_prompt_index = st.selectbox(
        "Select Prompt to Delete", 
        range(len(prompts)), 
        format_func=lambda x: f"{prompts[x].get('agent_name', 'Unnamed')}"
    )
    
    selected_prompt = prompts[selected_prompt_index]
    
    # Confirmation before deletion
    if st.button("Confirm Delete"):
        try:
            api_url = f"{st.secrets.get('API_BASE_URL', 'http://localhost:8000')}/db/delete_rows/prompts"
            
            # Prepare delete conditions
            delete_conditions = {
                "id": selected_prompt.get('id'),
                "user_id": st.session_state.username_id
            }
            
            response = requests.post(api_url, json=delete_conditions)
            
            if response.status_code == 200:
                st.success("Prompt deleted successfully!")
                st.rerun()  # Replace st.experimental_rerun() with st.rerun()
            else:
                st.error(f"Failed to delete prompt: {response.text}")
        
        except Exception as e:
            st.error(f"Error deleting prompt: {e}")

def render_prompts_page():
    st.title("Prompts Manager")

    # Ensure user is logged in
    if not st.session_state.get("username_id"):
        st.warning("Please log in to manage prompts")
        return

    # Add custom CSS (keep existing CSS)
    st.markdown("""
        <style>
        /* Existing CSS styles */
        </style>
    """, unsafe_allow_html=True)

    # Create tabs for different prompt management actions
    tab1, tab2, tab3, tab4 = st.tabs([
        "View Prompts", 
        "Add New Prompt", 
        "Edit Prompt", 
        "Delete Prompt"
    ])

    with tab1:
        st.subheader("Your Prompts")
        prompts = fetch_user_prompts()
        
        if prompts:
            # Display prompts in a more readable format
            for prompt in prompts:
                with st.expander(f"Prompt: {prompt.get('agent_name', 'Unnamed')}"):
                    st.json(prompt)
        else:
            st.info("No prompts found. Add a new prompt to get started!")

    with tab2:
        add_new_prompt()

    with tab3:
        edit_prompt()

    with tab4:
        delete_prompt()

if __name__ == "__main__":
    render_prompts_page() 