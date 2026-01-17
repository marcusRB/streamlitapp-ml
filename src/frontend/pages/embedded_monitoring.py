"""
Enhanced Monitoring Page with Embedded Views
View FastAPI Swagger UI and MLflow UI directly in Streamlit
"""

import streamlit as st
import streamlit.components.v1 as components
import requests
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from frontend.components.embedded_docs import (
    embed_fastapi_docs,
    embed_mlflow_ui,
    create_service_card
)

# Page config
st.set_page_config(
    page_title="Embedded Monitoring",
    page_icon="📺",
    layout="wide"
)

# Title
st.title("📺 Embedded Service Monitoring")
st.markdown("View FastAPI documentation and MLflow UI directly in Streamlit")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Service Configuration")
    
    backend_url = st.text_input(
        "Backend API URL",
        value="http://localhost:8000",
        help="URL of the FastAPI backend"
    )
    
    mlflow_url = st.text_input(
        "MLflow Server URL",
        value="http://localhost:5000",
        help="URL of the MLflow tracking server"
    )
    
    st.divider()
    
    iframe_height = st.slider(
        "View Height (px)",
        400, 1200, 800, 50,
        help="Adjust the height of embedded views"
    )
    
    st.divider()
    
    st.subheader("🔗 Quick Links")
    st.markdown(f"- [API Docs]({backend_url}/docs)")
    st.markdown(f"- [API Health]({backend_url}/health)")
    st.markdown(f"- [MLflow UI]({mlflow_url})")

# Check service status
def check_service_status(url: str, timeout: int = 5) -> tuple:
    """Check if service is online"""
    try:
        response = requests.get(f"{url}/health", timeout=timeout)
        if response.status_code == 200:
            return ("online", response.json())
        else:
            return ("error", f"Status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        return ("offline", "Connection refused")
    except requests.exceptions.Timeout:
        return ("offline", "Timeout")
    except Exception as e:
        return ("error", str(e))

# Service Status Overview
st.header("📊 Service Status Overview")

col1, col2 = st.columns(2)

with col1:
    with st.spinner("Checking Backend API..."):
        backend_status, backend_info = check_service_status(backend_url)
    
    create_service_card(
        "FastAPI Backend",
        backend_url,
        backend_status,
        "REST API for CKD predictions and model serving",
        "🚀"
    )

with col2:
    with st.spinner("Checking MLflow Server..."):
        mlflow_status, mlflow_info = check_service_status(mlflow_url)
    
    create_service_card(
        "MLflow Tracking Server",
        mlflow_url,
        mlflow_status,
        "Experiment tracking and model registry",
        "🧪"
    )

st.divider()

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📚 API Documentation",
    "🧪 MLflow Experiments", 
    "📊 MLflow Dashboard",
    "🔧 Custom View"
])

# ============================================================================
# TAB 1: FastAPI Documentation
# ============================================================================
with tab1:
    if backend_status == "online":
        st.info("💡 **Tip:** You can test API endpoints directly using the 'Try it out' button in Swagger UI")
        embed_fastapi_docs(backend_url, height=iframe_height)
    else:
        st.error("❌ Backend API is not available")
        st.info(f"**Status:** {backend_info}")
        st.markdown("""
        **To start the backend:**
        ```bash
        # Terminal 1
        uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
        
        # Or using Docker
        docker-compose up backend
        ```
        """)

# ============================================================================
# TAB 2: MLflow Experiments View
# ============================================================================
with tab2:
    if mlflow_status == "online":
        st.info("💡 **Tip:** Click on any run to view detailed metrics, parameters, and artifacts")
        embed_mlflow_ui(mlflow_url, height=iframe_height)
    else:
        st.error("❌ MLflow Server is not available")
        st.info(f"**Status:** {mlflow_info}")
        st.markdown("""
        **To start MLflow:**
        ```bash
        # Terminal 1
        mlflow ui --host 0.0.0.0 --port 5000
        
        # Or using Docker
        docker-compose up mlflow
        ```
        
        **Common Issues:**
        1. Port 5000 already in use → Kill the process: `lsof -ti:5000 | xargs kill -9`
        2. Can't connect → Check if MLflow is binding to 0.0.0.0, not 127.0.0.1
        3. Docker issue → Check logs: `docker logs ckd_mlflow_server`
        """)

# ============================================================================
# TAB 3: MLflow Dashboard View
# ============================================================================
with tab3:
    st.header("📊 MLflow Full Dashboard")
    
    if mlflow_status == "online":
        st.info("💡 **Tip:** This is the complete MLflow UI. Navigate using the left sidebar in the embedded view.")
        
        # Full MLflow UI
        mlflow_full_url = mlflow_url
        
        st.caption(f"Displaying: {mlflow_full_url}")
        
        iframe_code = f"""
        <iframe 
            src="{mlflow_full_url}" 
            width="100%" 
            height="{iframe_height}px" 
            frameborder="0"
            style="border: 2px solid #e0e0e0; border-radius: 5px;"
        ></iframe>
        """
        
        components.html(iframe_code, height=iframe_height, scrolling=True)
    else:
        st.error("❌ MLflow Server is not available")

# ============================================================================
# TAB 4: Custom View
# ============================================================================
with tab4:
    st.header("🔧 Custom URL Viewer")
    st.markdown("View any URL in an embedded iframe")
    
    custom_url = st.text_input(
        "Enter URL",
        value=backend_url,
        placeholder="http://localhost:8000",
        help="Enter any URL to display it in an iframe"
    )
    
    custom_height = st.slider(
        "Height (px)",
        400, 1200, 600, 50,
        key="custom_height"
    )
    
    if st.button("Load URL", type="primary"):
        if custom_url:
            st.info(f"Loading: {custom_url}")
            
            iframe_code = f"""
            <iframe 
                src="{custom_url}" 
                width="100%" 
                height="{custom_height}px" 
                frameborder="0"
                style="border: 2px solid #e0e0e0; border-radius: 5px;"
            ></iframe>
            """
            
            components.html(iframe_code, height=custom_height, scrolling=True)
        else:
            st.warning("Please enter a URL")

# ============================================================================
# Footer
# ============================================================================
st.divider()

st.subheader("ℹ️ Usage Tips")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **FastAPI Swagger UI:**
    - Click "Try it out" to test endpoints
    - View request/response schemas
    - Execute API calls directly
    - Download OpenAPI specification
    """)

with col2:
    st.markdown("""
    **MLflow UI:**
    - Compare multiple runs
    - View metric charts
    - Download artifacts
    - Register models
    """)

# Troubleshooting section
with st.expander("🔍 Troubleshooting"):
    st.markdown("""
    ### Services Not Loading?
    
    **1. Check if services are running:**
    ```bash
    # Check backend
    curl http://localhost:8000/health
    
    # Check MLflow
    curl http://localhost:5000/health
    ```
    
    **2. Check Docker containers:**
    ```bash
    docker ps
    docker logs ckd_backend_api
    docker logs ckd_mlflow_server
    ```
    
    **3. Port conflicts:**
    ```bash
    # Find what's using the port
    lsof -i :8000
    lsof -i :5000
    
    # Kill process if needed
    lsof -ti:5000 | xargs kill -9
    ```
    
    **4. Docker network issues:**
    ```bash
    # Restart Docker Compose
    docker-compose down
    docker-compose up -d
    
    # Check network
    docker network inspect ckd_network
    ```
    
    **5. Browser issues:**
    - Try opening in incognito mode
    - Clear browser cache
    - Try different browser
    - Check browser console for errors (F12)
    
    **6. CORS issues:**
    - Ensure backend CORS allows Streamlit origin
    - Check backend logs for CORS errors
    """)

# Auto-refresh option
if st.sidebar.checkbox("🔄 Auto-refresh status"):
    import time
    time.sleep(30)
    st.rerun()