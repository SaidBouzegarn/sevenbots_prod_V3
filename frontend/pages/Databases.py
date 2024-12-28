import streamlit as st
import requests
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def fetch_available_tables():
    """
    Fetch available tables from the backend.
    """
    try:
        response = requests.get("http://localhost:8000/db/tables")
        if response.status_code == 200:
            return response.json()  # Returns a list of table names
        else:
            st.error(f"Failed to fetch tables: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error fetching tables: {e}")
        return []

def fetch_table_data(table_name, conditions=None):
    """
    Fetch table data using POST request to /db/query/{table_name}
    """
    try:
        # Ensure user is logged in
        if not st.session_state.get("username_id"):
            st.warning("Please log in to view database")
            return {"rows": [], "total_count": 0}

        # Prepare payload
        payload = {
            "conditions": conditions or {},
            # Optionally add user_id condition if needed
            # "conditions": {"user_id": st.session_state.username_id}
        }

        # Make the API request
        response = requests.post(
            f"{st.secrets.get('API_BASE_URL', 'http://localhost:8000')}/db/query/{table_name}", 
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "rows": data,
                "total_count": len(data)
            }
        else:
            st.error(f"Failed to fetch table data: {response.text}")
            return {"rows": [], "total_count": 0}
    
    except Exception as e:
        st.error(f"Error fetching table data: {e}")
        return {"rows": [], "total_count": 0}

def update_table_data(table_name, rows):
    """
    Update or add rows to a table
    """
    try:
        # Ensure user is logged in
        if not st.session_state.get("username_id"):
            st.warning("Please log in to update database")
            return {"status": "error", "detail": "Not logged in"}

        # Determine the appropriate endpoint based on the action
        api_url = f"{st.secrets.get('API_BASE_URL', 'http://localhost:8000')}/db/add_rows/{table_name}"
        
        # Make the API request
        response = requests.post(
            api_url, 
            json=rows
        )
        
        if response.status_code == 200:
            return {"status": "success"}
        else:
            return {
                "status": "error",
                "detail": response.text
            }
    
    except Exception as e:
        return {
            "status": "error",
            "detail": str(e)
        }

def render_databases_page():
    st.title("Database Manager")

    # Add custom CSS
    st.markdown("""
        <style>
        .stButton button {
            background: linear-gradient(120deg, #000000, #1e3a8a);
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 5px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stSelectbox {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
        }
        
        .dataframe {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
            padding: 10px;
        }
        
        .info-box {
            padding: 1rem;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Ensure user is logged in
    if not st.session_state.get("username_id"):
        st.warning("Please log in to access Database Manager")
        return

    # Add info box about database
    st.markdown("""
        <div class="info-box">
        This page allows you to view and edit database tables. 
        Be cautious when making changes as they directly affect the database.
        </div>
    """, unsafe_allow_html=True)

    # Fetch available tables
    tables = fetch_available_tables()
    if not tables:
        st.warning("No tables found")
        return

    # Table selection
    selected_table = st.selectbox("Select Table", options=tables)

    # Fetch and display data
    result = fetch_table_data(selected_table)
    if not result['rows']:
        st.warning("No data found")
        return

    # Convert to DataFrame
    df = pd.DataFrame(result['rows'])

    # Editable data
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True
    )

    # Save changes button
    if st.button("Save Changes"):
        # Convert edited DataFrame to list of dictionaries
        rows_to_update = edited_df.to_dict('records')
        
        # Update via backend
        update_result = update_table_data(selected_table, rows_to_update)
        
        if update_result.get('status') == 'success':
            st.success("Successfully added/updated rows!")
            # Refresh the data
            st.rerun()
        else:
            error_msg = update_result.get('detail', 'Failed to update rows')
            st.error(f"Update failed: {error_msg}")

if __name__ == "__main__":
    render_databases_page() 