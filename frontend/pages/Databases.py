import streamlit as st
import aiohttp
import asyncio
import pandas as pd
import json

async def fetch_available_tables():
    """
    Fetch available tables (adapted to match /db/tables endpoint with GET).
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/db/tables") as response:
                response.raise_for_status()
                return await response.json()  # Returns a list of table names
    except Exception as e:
        st.error(f"Error fetching tables: {e}")
        return []

async def fetch_table_data(table_name, conditions=None):
    """
    Fetch table data by POSTing to /db/query/{table_name}
    using json=payload (like test_query_table in db_service_test.py).
    """
    try:
        payload = {
            "conditions": conditions or {}
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://localhost:8000/db/query/{table_name}",
                json=payload
            ) as response:
                response.raise_for_status()
                data = await response.json()
                # The /db/query endpoint returns a list of rows.
                return {
                    "rows": data,
                    "total_count": len(data)
                }
    except aiohttp.ClientResponseError as e:
        st.error(f"Network error fetching table data: {e}")
        try:
            error_text = await e.text()
            st.error(f"Response content: {error_text or 'No response text'}")
        except:
            pass
        return {"rows": [], "total_count": 0}
    except ValueError as e:
        st.error(f"JSON parsing error: {e}")
        return {"rows": [], "total_count": 0}

async def update_table_data(table_name, rows):
    """
    Post new/updated rows to /db/add_rows/{table_name}.
    Per test_add_rows_endpoint in db_service_test.py, we send content=json.dumps(...).
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://localhost:8000/db/add_rows/{table_name}",
                content=json.dumps(rows),
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    return await response.json()  # e.g., {"status": "success"}
                else:
                    detail = await response.json()
                    return {
                        "status": "error",
                        "detail": detail
                    }
    except aiohttp.ClientError as e:
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

    # Add info box about news_scrapper.db
    st.markdown("""
        <div class="info-box">
        This page allows you to view and edit the news scrapper database tables. 
        The database contains information about scraped articles and website configurations.
        </div>
    """, unsafe_allow_html=True)

    # Fetch available tables asynchronously
    tables = asyncio.run(fetch_available_tables())
    if not tables:
        st.warning("No tables found")
        return

    # Table selection
    selected_table = st.selectbox("Select Table", options=tables)

    # Fetch and display data asynchronously
    result = asyncio.run(fetch_table_data(selected_table))
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
        rows_to_update = edited_df.to_dict('records')
        # Update via backend asynchronously
        update_result = asyncio.run(update_table_data(selected_table, rows_to_update))
        
        if update_result.get('status') == 'success':
            st.success("Successfully added/updated rows!")
        else:
            error_msg = update_result.get('detail', 'Failed to update rows')
            st.error(f"Update failed: {error_msg}")

if __name__ == "__main__":
    render_databases_page() 