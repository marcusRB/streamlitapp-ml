"""
API & MLflow Monitoring Page
View FastAPI endpoints and MLflow experiments in Streamlit
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from frontend.utils.api_client import get_api_client

# Page config
st.set_page_config(
    page_title="API & MLflow Monitor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 API & MLflow Monitoring Dashboard")
st.markdown("Monitor FastAPI endpoints, MLflow experiments, and system health")

# Initialize API client
try:
    api_client = get_api_client(base_url="http://backend:8000")
except:
    api_client = get_api_client(base_url="http://localhost:8000")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    backend_url = st.text_input(
        "Backend URL",
        value="http://localhost:8000",
        help="URL of the FastAPI backend"
    )
    
    mlflow_url = st.text_input(
        "MLflow URL",
        value="http://localhost:5000",
        help="URL of the MLflow server"
    )
    
    st.divider()
    
    auto_refresh = st.checkbox("Auto Refresh", value=False)
    if auto_refresh:
        refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 10)
    
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.rerun()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🏥 System Health", 
    "🔌 API Endpoints", 
    "🧪 MLflow Experiments",
    "📈 MLflow Runs"
])

# ============================================================================
# TAB 1: System Health
# ============================================================================
with tab1:
    st.header("System Health Status")
    
    col1, col2, col3 = st.columns(3)
    
    # Check Backend API
    with col1:
        st.subheader("FastAPI Backend")
        try:
            response = requests.get(f"{backend_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                st.success("✅ **Online**")
                st.metric("Status", health_data.get('status', 'unknown').upper())
                st.metric("Models Loaded", len(health_data.get('models_loaded', [])))
                
                with st.expander("Details"):
                    st.json(health_data)
            else:
                st.error(f"❌ **Error** (Status: {response.status_code})")
        except requests.exceptions.ConnectionError:
            st.error("❌ **Offline**")
            st.caption("Cannot connect to backend")
        except Exception as e:
            st.error(f"❌ **Error**: {str(e)}")
    
    # Check MLflow
    with col2:
        st.subheader("MLflow Server")
        try:
            response = requests.get(f"{mlflow_url}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ **Online**")
                st.metric("Status", "HEALTHY")
                st.caption(f"🔗 [Open MLflow UI]({mlflow_url})")
            else:
                st.error(f"❌ **Error** (Status: {response.status_code})")
        except requests.exceptions.ConnectionError:
            st.error("❌ **Offline**")
            st.caption("Cannot connect to MLflow")
            st.info(f"💡 Try: `mlflow ui --host 0.0.0.0 --port 5000`")
        except Exception as e:
            st.error(f"❌ **Error**: {str(e)}")
    
    # Check Streamlit (self)
    with col3:
        st.subheader("Streamlit Frontend")
        st.success("✅ **Online**")
        st.metric("Status", "RUNNING")
        st.caption("You're viewing it now!")
    
    st.divider()
    
    # System Information
    st.subheader("📋 System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Backend API**")
        st.code(backend_url)
        st.write("**API Documentation**")
        st.markdown(f"- [Swagger UI]({backend_url}/docs)")
        st.markdown(f"- [ReDoc]({backend_url}/redoc)")
    
    with col2:
        st.write("**MLflow Server**")
        st.code(mlflow_url)
        st.write("**MLflow UI**")
        st.markdown(f"- [Experiments]({mlflow_url})")
        st.markdown(f"- [Models]({mlflow_url}/#/models)")

# ============================================================================
# TAB 2: API Endpoints
# ============================================================================
with tab2:
    st.header("FastAPI Endpoints")
    
    try:
        # Get available models
        response = requests.get(f"{backend_url}/models", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            
            st.subheader("🤖 Available Models")
            
            models_list = models_data.get('models', {})
            if models_list:
                models_df = pd.DataFrame([
                    {
                        'Model': name,
                        'Type': info.get('type', 'N/A'),
                        'Has Probability': '✅' if info.get('has_probability') else '❌'
                    }
                    for name, info in models_list.items()
                ])
                st.dataframe(models_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No models available")
        
        st.divider()
        
        # Interactive API Tester
        st.subheader("🧪 Test API Endpoints")
        
        endpoint = st.selectbox(
            "Select Endpoint",
            [
                "/health",
                "/models",
                "/predict",
                "/predict/ensemble"
            ]
        )
        
        if endpoint in ["/predict", "/predict/ensemble"]:
            st.write("**Patient Data:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                hemo = st.number_input("Hemoglobin", 0.0, 20.0, 15.4, 0.1)
                sg = st.number_input("Specific Gravity", 1.000, 1.030, 1.020, 0.001, format="%.3f")
                sc = st.number_input("Serum Creatinine", 0.0, 20.0, 1.2, 0.1)
            
            with col2:
                rbcc = st.number_input("RBC Count", 0.0, 10.0, 5.2, 0.1)
                pcv = st.number_input("PCV", 0.0, 60.0, 44.0, 1.0)
                htn = st.selectbox("Hypertension", [0, 1], index=1)
            
            with col3:
                dm = st.selectbox("Diabetes", [0, 1], index=1)
                bp = st.number_input("Blood Pressure", 0.0, 200.0, 80.0, 1.0)
                age = st.number_input("Age", 0, 120, 48, 1)
            
            patient_data = {
                'hemo': hemo, 'sg': sg, 'sc': sc,
                'rbcc': rbcc, 'pcv': pcv, 'htn': float(htn),
                'dm': float(dm), 'bp': bp, 'age': float(age)
            }
            
            if endpoint == "/predict":
                model_name = st.selectbox("Model", ["KNN", "SVM", "GradientBoosting", "HistGradientBoosting"])
            
            if st.button("📤 Send Request", use_container_width=True):
                try:
                    if endpoint == "/predict":
                        url = f"{backend_url}/predict?model_name={model_name}"
                        response = requests.post(url, json=patient_data, timeout=10)
                    else:
                        url = f"{backend_url}/predict/ensemble"
                        response = requests.post(url, json=patient_data, timeout=10)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Request Successful")
                        
                        if endpoint == "/predict":
                            col1, col2 = st.columns(2)
                            with col1:
                                pred = result['prediction'].upper()
                                if pred == "CKD":
                                    st.error(f"### 🔴 {pred}")
                                else:
                                    st.success(f"### 🟢 {pred}")
                            with col2:
                                st.metric("Confidence", f"{result['confidence']:.1%}")
                        else:
                            # Ensemble result
                            st.subheader("Individual Predictions")
                            for model, pred in result['individual_predictions'].items():
                                with st.expander(f"{model}"):
                                    st.write(f"Prediction: **{pred['prediction'].upper()}**")
                                    st.write(f"Confidence: **{pred['confidence']:.1%}**")
                            
                            st.subheader("Consensus")
                            consensus = result['consensus']
                            st.write(f"**Prediction:** {consensus['prediction'].upper()}")
                            st.write(f"**Agreement:** {consensus['agreement']}")
                            st.write(f"**Confidence:** {consensus['consensus_confidence']:.1%}")
                        
                        with st.expander("📄 Full Response"):
                            st.json(result)
                    else:
                        st.error(f"❌ Error: {response.status_code}")
                        st.code(response.text)
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        else:
            if st.button("📤 Send Request", use_container_width=True):
                try:
                    response = requests.get(f"{backend_url}{endpoint}", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ Request Successful")
                        st.json(response.json())
                    else:
                        st.error(f"❌ Error: {response.status_code}")
                        st.code(response.text)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    except Exception as e:
        st.error(f"Cannot connect to backend: {str(e)}")
        st.info("Make sure the backend is running at " + backend_url)

# ============================================================================
# TAB 3: MLflow Experiments
# ============================================================================
with tab3:
    st.header("MLflow Experiments")
    
    try:
        # Try to get experiments from backend API
        response = requests.get(f"{backend_url}/mlflow/experiments", timeout=5)
        
        if response.status_code == 200:
            experiments_data = response.json()
            experiments = experiments_data.get('experiments', [])
            
            if experiments:
                st.success(f"Found {len(experiments)} experiment(s)")
                
                for exp in experiments:
                    with st.expander(f"📊 {exp['name']}", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Experiment ID", exp['experiment_id'])
                        with col2:
                            st.metric("Lifecycle", exp['lifecycle_stage'].upper())
                        with col3:
                            st.write("**Artifact Location**")
                            st.code(exp['artifact_location'], language=None)
                        
                        # Link to MLflow UI
                        st.markdown(f"🔗 [View in MLflow UI]({mlflow_url}/#/experiments/{exp['experiment_id']})")
            else:
                st.info("No experiments found")
                st.caption("Train some models to see experiments here!")
        
        else:
            st.warning("Could not fetch experiments from API")
            st.info("Try accessing MLflow directly:")
            st.markdown(f"🔗 [Open MLflow UI]({mlflow_url})")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("**Troubleshooting:**")
        st.markdown(f"""
        1. Check MLflow is running: `{mlflow_url}`
        2. Check backend is running: `{backend_url}`
        3. Start MLflow: `mlflow ui --host 0.0.0.0 --port 5000`
        """)

# ============================================================================
# TAB 4: MLflow Runs
# ============================================================================
with tab4:
    st.header("MLflow Runs")
    
    # Experiment selector
    experiment_name = st.text_input(
        "Experiment Name",
        value="CKD_Detection",
        help="Enter the MLflow experiment name"
    )
    
    limit = st.slider("Number of Runs", 1, 50, 10)
    
    if st.button("📥 Fetch Runs", use_container_width=True):
        try:
            response = requests.get(
                f"{backend_url}/mlflow/runs/{experiment_name}",
                params={"limit": limit},
                timeout=10
            )
            
            if response.status_code == 200:
                runs_data = response.json()
                runs = runs_data.get('runs', [])
                
                if runs:
                    st.success(f"Found {len(runs)} run(s)")
                    
                    # Create dataframe
                    runs_list = []
                    for run in runs:
                        runs_list.append({
                            'Run Name': run.get('run_name', 'N/A'),
                            'Status': run.get('status', 'N/A'),
                            'F1 Score': run.get('metrics', {}).get('test_f1_score', 'N/A'),
                            'Accuracy': run.get('metrics', {}).get('test_accuracy', 'N/A'),
                            'Run ID': run.get('run_id', 'N/A')[:8] + '...'
                        })
                    
                    runs_df = pd.DataFrame(runs_list)
                    
                    # Display dataframe
                    st.dataframe(
                        runs_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Detailed view
                    st.subheader("Detailed Run Information")
                    
                    for i, run in enumerate(runs):
                        with st.expander(f"Run: {run.get('run_name', 'N/A')}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Metrics**")
                                metrics = run.get('metrics', {})
                                for key, value in metrics.items():
                                    if isinstance(value, float):
                                        st.metric(key, f"{value:.4f}")
                                    else:
                                        st.metric(key, value)
                            
                            with col2:
                                st.write("**Parameters**")
                                params = run.get('params', {})
                                for key, value in params.items():
                                    st.write(f"- **{key}**: {value}")
                            
                            st.write(f"**Run ID:** `{run.get('run_id')}`")
                            st.markdown(f"🔗 [View in MLflow]({mlflow_url}/#/experiments/{experiment_name}/runs/{run.get('run_id')})")
                else:
                    st.info(f"No runs found for experiment '{experiment_name}'")
            
            else:
                st.error(f"Error: {response.status_code}")
                st.code(response.text)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure the backend and MLflow are running")

# ============================================================================
# Footer
# ============================================================================
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📚 Quick Links")
    st.markdown(f"- [API Docs]({backend_url}/docs)")
    st.markdown(f"- [API ReDoc]({backend_url}/redoc)")

with col2:
    st.subheader("🧪 MLflow")
    st.markdown(f"- [MLflow UI]({mlflow_url})")
    st.markdown(f"- [Experiments]({mlflow_url}/#/experiments)")

with col3:
    st.subheader("ℹ️ Information")
    st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"**Backend:** {backend_url}")
    st.write(f"**MLflow:** {mlflow_url}")

# Auto refresh
if auto_refresh:
    import time
    time.sleep(refresh_interval)
    st.rerun()