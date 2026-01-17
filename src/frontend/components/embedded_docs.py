"""
Embedded Documentation Components
Display FastAPI docs and MLflow UI inside Streamlit using iframes
"""

import streamlit as st
import streamlit.components.v1 as components


def embed_fastapi_docs(backend_url: str = "http://localhost:8000", height: int = 800):
    """
    Embed FastAPI Swagger UI in Streamlit
    
    Args:
        backend_url: URL of the FastAPI backend
        height: Height of the iframe in pixels
    """
    st.subheader("📚 FastAPI Interactive Documentation")
    
    # Tab selection
    doc_type = st.radio(
        "Documentation Type",
        ["Swagger UI", "ReDoc"],
        horizontal=True
    )
    
    if doc_type == "Swagger UI":
        docs_url = f"{backend_url}/docs"
        st.caption("Interactive API documentation with try-it-out functionality")
    else:
        docs_url = f"{backend_url}/redoc"
        st.caption("Clean, three-panel API documentation")
    
    # Display URL
    st.info(f"📍 **Direct Link:** [{docs_url}]({docs_url})")
    
    # Embed iframe
    iframe_code = f"""
    <iframe 
        src="{docs_url}" 
        width="100%" 
        height="{height}px" 
        frameborder="0"
        style="border: 2px solid #e0e0e0; border-radius: 5px;"
    ></iframe>
    """
    
    components.html(iframe_code, height=height, scrolling=True)


def embed_mlflow_ui(mlflow_url: str = "http://localhost:5000", height: int = 800):
    """
    Embed MLflow UI in Streamlit
    
    Args:
        mlflow_url: URL of the MLflow server
        height: Height of the iframe in pixels
    """
    st.subheader("🧪 MLflow Tracking UI")
    
    # Tab selection
    view = st.selectbox(
        "MLflow View",
        ["Experiments", "Models", "Specific Experiment"],
        help="Select which MLflow view to display"
    )
    
    if view == "Experiments":
        target_url = f"{mlflow_url}/#/experiments"
        st.caption("View all experiments and their runs")
    elif view == "Models":
        target_url = f"{mlflow_url}/#/models"
        st.caption("View registered models and their versions")
    else:
        exp_id = st.text_input(
            "Experiment ID", 
            value="0",
            help="Enter the experiment ID to view"
        )
        target_url = f"{mlflow_url}/#/experiments/{exp_id}"
        st.caption(f"View experiment {exp_id} details")
    
    # Display URL
    st.info(f"📍 **Direct Link:** [{target_url}]({target_url})")
    
    # Embed iframe
    iframe_code = f"""
    <iframe 
        src="{target_url}" 
        width="100%" 
        height="{height}px" 
        frameborder="0"
        style="border: 2px solid #e0e0e0; border-radius: 5px;"
    ></iframe>
    """
    
    components.html(iframe_code, height=height, scrolling=True)


def embed_custom_url(url: str, title: str = "External Content", height: int = 800):
    """
    Embed any URL in Streamlit
    
    Args:
        url: URL to embed
        title: Title for the embedded content
        height: Height of the iframe in pixels
    """
    st.subheader(title)
    st.info(f"📍 **Direct Link:** [{url}]({url})")
    
    iframe_code = f"""
    <iframe 
        src="{url}" 
        width="100%" 
        height="{height}px" 
        frameborder="0"
        style="border: 2px solid #e0e0e0; border-radius: 5px;"
    ></iframe>
    """
    
    components.html(iframe_code, height=height, scrolling=True)


def create_service_card(
    service_name: str,
    url: str,
    status: str,
    description: str,
    icon: str = "🔵"
):
    """
    Create a service status card
    
    Args:
        service_name: Name of the service
        url: URL of the service
        status: Status of the service (online/offline)
        description: Service description
        icon: Emoji icon for the service
    """
    status_color = "green" if status.lower() == "online" else "red"
    status_icon = "✅" if status.lower() == "online" else "❌"
    
    card_html = f"""
    <div style="
        border: 2px solid #{status_color if status.lower() == 'online' else 'ff0000'}40;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    ">
        <h3 style="margin: 0 0 10px 0;">
            {icon} {service_name} {status_icon}
        </h3>
        <p style="margin: 5px 0; color: #666;">
            {description}
        </p>
        <p style="margin: 10px 0 0 0;">
            <strong>URL:</strong> 
            <a href="{url}" target="_blank" style="color: #1f77b4;">
                {url}
            </a>
        </p>
        <p style="margin: 5px 0 0 0;">
            <strong>Status:</strong> 
            <span style="color: {status_color}; font-weight: bold;">
                {status.upper()}
            </span>
        </p>
    </div>
    """
    
    components.html(card_html, height=200)


# Example usage in Streamlit page
if __name__ == "__main__":
    st.set_page_config(page_title="Embedded Docs Demo", layout="wide")
    
    st.title("📚 Embedded Documentation Demo")
    
    # Configuration
    backend_url = st.sidebar.text_input("Backend URL", "http://localhost:8000")
    mlflow_url = st.sidebar.text_input("MLflow URL", "http://localhost:5000")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["FastAPI Docs", "MLflow UI", "Service Cards"])
    
    with tab1:
        embed_fastapi_docs(backend_url)
    
    with tab2:
        embed_mlflow_ui(mlflow_url)
    
    with tab3:
        st.header("Service Status Cards")
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_service_card(
                "FastAPI Backend",
                backend_url,
                "online",
                "REST API for CKD predictions",
                "🚀"
            )
        
        with col2:
            create_service_card(
                "MLflow Server",
                mlflow_url,
                "online",
                "Experiment tracking and model registry",
                "🧪"
            )