"""
SHAP Explainability Page
Model interpretability using SHAP (SHapley Additive exPlanations)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import sys
from pathlib import Path
import joblib

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Page config
st.set_page_config(
    page_title="SHAP Explainability",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'shap_computed' not in st.session_state:
    st.session_state.shap_computed = False
if 'explainer' not in st.session_state:
    st.session_state.explainer = None
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None

# Title
st.title("🔍 Model Explainability with SHAP")
st.markdown("Understand **why** the model makes specific predictions")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model_name = st.selectbox(
        "Select Model",
        ["KNN", "SVM", "GradientBoosting", "HistGradientBoosting"],
        help="Choose which model to explain"
    )
    
    # Data selection
    data_source = st.radio(
        "Data Source",
        ["Test Set", "Upload CSV", "Single Instance"],
        help="Where to get data for explanation"
    )
    
    st.divider()
    
    # SHAP settings
    st.subheader("SHAP Settings")
    
    num_samples = st.slider(
        "Number of Samples",
        10, 100, 50,
        help="How many samples to explain"
    )
    
    background_samples = st.slider(
        "Background Samples",
        10, 200, 100,
        help="Samples for SHAP background distribution"
    )
    
    st.divider()
    
    compute_button = st.button(
        "🔬 Compute SHAP Values",
        type="primary",
        use_container_width=True
    )

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model(model_name: str):
    """Load trained model"""
    try:
        model_path = settings.get_model_path(model_name)
        model = joblib.load(model_path)
        logger.info(f"Loaded model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_test_data():
    """Load test dataset"""
    try:
        # Try normalized data first
        data_path = settings.paths.NORMALIZED_DATA_FILE
        if not data_path.exists():
            data_path = settings.paths.IMPUTED_DATA_FILE
        
        df = pd.read_csv(data_path)
        
        # Separate features and target
        X = df[settings.model.FEATURE_NAMES]
        y = df['status'] if 'status' in df.columns else None
        
        return X, y
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None, None

def compute_shap_values(model, X_train, X_test, model_name):
    """Compute SHAP values for the model"""
    try:
        # Select appropriate explainer based on model type
        if model_name in ["GradientBoosting", "HistGradientBoosting"]:
            # Tree explainer for tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        else:
            # Kernel explainer for other models
            background = shap.sample(X_train, background_samples)
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_test)
        
        return explainer, shap_values
    except Exception as e:
        logger.error(f"Error computing SHAP values: {str(e)}")
        return None, None

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Load model
model = load_model(model_name)

if model is None:
    st.error(f"❌ Could not load model: {model_name}")
    st.info("Please train the model first using the Model Training page")
    st.stop()

st.success(f"✅ Model loaded: {model_name}")

# Load or prepare data
if data_source == "Test Set":
    X, y = load_test_data()
    
    if X is None:
        st.error("❌ Could not load test data")
        st.stop()
    
    # Sample data
    if len(X) > num_samples:
        X_sample = X.sample(n=num_samples, random_state=42)
        y_sample = y.loc[X_sample.index] if y is not None else None
    else:
        X_sample = X
        y_sample = y
    
    st.info(f"📊 Using {len(X_sample)} samples from test set")

elif data_source == "Upload CSV":
    uploaded_file = st.file_uploader(
        "Upload CSV with patient data",
        type=['csv'],
        help="CSV should have columns: " + ", ".join(settings.model.FEATURE_NAMES)
    )
    
    if uploaded_file is None:
        st.warning("⏳ Please upload a CSV file")
        st.stop()
    
    X_sample = pd.read_csv(uploaded_file)
    y_sample = None
    
    # Validate columns
    missing_cols = set(settings.model.FEATURE_NAMES) - set(X_sample.columns)
    if missing_cols:
        st.error(f"❌ Missing columns: {missing_cols}")
        st.stop()
    
    X_sample = X_sample[settings.model.FEATURE_NAMES]
    st.success(f"✅ Loaded {len(X_sample)} samples from uploaded file")

else:  # Single Instance
    st.subheader("Enter Patient Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hemo = st.number_input("Hemoglobin (g/dL)", 0.0, 20.0, 15.4, 0.1)
        sg = st.number_input("Specific Gravity", 1.000, 1.030, 1.020, 0.001, format="%.3f")
        sc = st.number_input("Serum Creatinine (mg/dL)", 0.0, 20.0, 1.2, 0.1)
    
    with col2:
        rbcc = st.number_input("RBC Count (millions/cmm)", 0.0, 10.0, 5.2, 0.1)
        pcv = st.number_input("PCV (%)", 0.0, 60.0, 44.0, 1.0)
        htn = st.selectbox("Hypertension", [0, 1], index=1)
    
    with col3:
        dm = st.selectbox("Diabetes Mellitus", [0, 1], index=1)
        bp = st.number_input("Blood Pressure (mmHg)", 0.0, 200.0, 80.0, 1.0)
        age = st.number_input("Age (years)", 0, 120, 48, 1)
    
    patient_data = {
        'hemo': hemo, 'sg': sg, 'sc': sc,
        'rbcc': rbcc, 'pcv': pcv, 'htn': float(htn),
        'dm': float(dm), 'bp': bp, 'age': float(age)
    }
    
    X_sample = pd.DataFrame([patient_data])
    y_sample = None

# Display data preview
with st.expander("📋 Data Preview", expanded=False):
    st.dataframe(X_sample.head(10), use_container_width=True)

# Compute SHAP values
if compute_button:
    with st.spinner("🔬 Computing SHAP values... This may take a moment..."):
        # Get background data
        X_background, _ = load_test_data()
        
        if X_background is None:
            st.error("❌ Could not load background data")
            st.stop()
        
        # Compute SHAP
        explainer, shap_values = compute_shap_values(
            model, 
            X_background, 
            X_sample,
            model_name
        )
        
        if explainer is None or shap_values is None:
            st.error("❌ Failed to compute SHAP values")
            st.stop()
        
        # Store in session state
        st.session_state.explainer = explainer
        st.session_state.shap_values = shap_values
        st.session_state.X_sample = X_sample
        st.session_state.y_sample = y_sample
        st.session_state.shap_computed = True
        
        st.success("✅ SHAP values computed successfully!")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

if st.session_state.shap_computed:
    st.divider()
    st.header("📊 SHAP Visualizations")
    
    shap_values = st.session_state.shap_values
    X_sample = st.session_state.X_sample
    explainer = st.session_state.explainer
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Summary Plot",
        "Force Plot",
        "Waterfall Plot",
        "Dependence Plot",
        "Feature Importance"
    ])
    
    # TAB 1: Summary Plot
    with tab1:
        st.subheader("📊 SHAP Summary Plot")
        st.markdown("Shows the impact of each feature on model predictions")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Handle both binary and multi-class
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[1], X_sample, show=False)
            else:
                shap.summary_plot(shap_values, X_sample, show=False)
            
            st.pyplot(fig)
            plt.close()
            
            st.info("""
            **How to read this plot:**
            - Each point represents a sample
            - Color indicates feature value (red=high, blue=low)
            - X-axis shows SHAP value (impact on prediction)
            - Features are ordered by importance (top to bottom)
            """)
        
        except Exception as e:
            st.error(f"Error creating summary plot: {str(e)}")
    
    # TAB 2: Force Plot
    with tab2:
        st.subheader("🎯 SHAP Force Plot")
        st.markdown("Shows how features push the prediction higher or lower")
        
        instance_idx = st.selectbox(
            "Select Instance",
            range(len(X_sample)),
            format_func=lambda x: f"Instance {x+1}"
        )
        
        try:
            # Get prediction
            prediction = model.predict(X_sample.iloc[[instance_idx]])[0]
            pred_label = "CKD" if prediction == 1 else "Not CKD"
            
            st.info(f"**Prediction for Instance {instance_idx+1}:** {pred_label}")
            
            # Create force plot
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][instance_idx]
            else:
                shap_vals = shap_values[instance_idx]
            
            # Display as matplotlib (streamlit-friendly)
            fig, ax = plt.subplots(figsize=(12, 3))
            
            shap.force_plot(
                explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
                shap_vals,
                X_sample.iloc[instance_idx],
                matplotlib=True,
                show=False
            )
            
            st.pyplot(fig)
            plt.close()
            
            st.info("""
            **How to read this plot:**
            - Red features push prediction towards CKD
            - Blue features push prediction towards Not CKD
            - Larger bars = stronger impact
            """)
        
        except Exception as e:
            st.error(f"Error creating force plot: {str(e)}")
            st.write("Showing waterfall plot instead...")
            
            # Fallback to waterfall
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_vals,
                        base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
                        data=X_sample.iloc[instance_idx].values,
                        feature_names=X_sample.columns.tolist()
                    ),
                    show=False
                )
                st.pyplot(fig)
                plt.close()
            except Exception as e2:
                st.error(f"Error: {str(e2)}")
    
    # TAB 3: Waterfall Plot
    with tab3:
        st.subheader("💧 SHAP Waterfall Plot")
        st.markdown("Step-by-step breakdown of how features contribute to prediction")
        
        instance_idx = st.selectbox(
            "Select Instance",
            range(len(X_sample)),
            format_func=lambda x: f"Instance {x+1}",
            key="waterfall_instance"
        )
        
        try:
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][instance_idx]
                expected_val = explainer.expected_value[1]
            else:
                shap_vals = shap_values[instance_idx]
                expected_val = explainer.expected_value
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals,
                    base_values=expected_val,
                    data=X_sample.iloc[instance_idx].values,
                    feature_names=X_sample.columns.tolist()
                ),
                show=False
            )
            
            st.pyplot(fig)
            plt.close()
            
            st.info("""
            **How to read this plot:**
            - Starts from base value (average prediction)
            - Each bar shows how a feature changes the prediction
            - Final value is the model's prediction for this instance
            """)
        
        except Exception as e:
            st.error(f"Error creating waterfall plot: {str(e)}")
    
    # TAB 4: Dependence Plot
    with tab4:
        st.subheader("📈 SHAP Dependence Plot")
        st.markdown("Shows how a feature's value affects predictions")
        
        feature = st.selectbox(
            "Select Feature",
            X_sample.columns.tolist(),
            help="Feature to analyze"
        )
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if isinstance(shap_values, list):
                shap.dependence_plot(
                    feature,
                    shap_values[1],
                    X_sample,
                    show=False
                )
            else:
                shap.dependence_plot(
                    feature,
                    shap_values,
                    X_sample,
                    show=False
                )
            
            st.pyplot(fig)
            plt.close()
            
            st.info(f"""
            **How to read this plot:**
            - X-axis: {feature} value
            - Y-axis: SHAP value (impact on prediction)
            - Each point is a sample
            - Color shows interaction with another feature
            """)
        
        except Exception as e:
            st.error(f"Error creating dependence plot: {str(e)}")
    
    # TAB 5: Feature Importance
    with tab5:
        st.subheader("⭐ Feature Importance")
        st.markdown("Ranking of features by their average impact on predictions")
        
        try:
            # Calculate mean absolute SHAP values
            if isinstance(shap_values, list):
                mean_shap = np.abs(shap_values[1]).mean(axis=0)
            else:
                mean_shap = np.abs(shap_values).mean(axis=0)
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'Feature': X_sample.columns,
                'Importance': mean_shap
            }).sort_values('Importance', ascending=False)
            
            # Bar plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title('Feature Importance (SHAP)')
            ax.invert_yaxis()
            plt.tight_layout()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.dataframe(
                    importance_df.style.background_gradient(subset=['Importance'], cmap='Blues'),
                    use_container_width=True,
                    hide_index=True
                )
            
            # Top 3 features
            st.success("**🏆 Top 3 Most Important Features:**")
            for i, row in importance_df.head(3).iterrows():
                st.write(f"{row['Feature']}: {row['Importance']:.4f}")
        
        except Exception as e:
            st.error(f"Error creating importance plot: {str(e)}")

else:
    # Show instructions
    st.info("👈 Click **'Compute SHAP Values'** in the sidebar to start")
    
    st.markdown("""
    ### What is SHAP?
    
    **SHAP (SHapley Additive exPlanations)** is a unified approach to explain predictions of machine learning models.
    
    #### Why use SHAP?
    - 🎯 **Understand predictions**: See which features influenced each prediction
    - 🔍 **Debug models**: Identify potential issues or biases
    - 📊 **Build trust**: Explain model decisions to stakeholders
    - ⚖️ **Fair ML**: Ensure models make decisions for the right reasons
    
    #### Available Visualizations:
    1. **Summary Plot**: Overall feature importance across all samples
    2. **Force Plot**: Individual prediction explanation
    3. **Waterfall Plot**: Step-by-step feature contribution
    4. **Dependence Plot**: How feature values affect predictions
    5. **Feature Importance**: Ranking of most impactful features
    """)
    
    # Example visualization
    st.markdown("### 📚 Example: What You'll See")
    
    st.image("https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/boston_instance.png", 
             caption="Example: SHAP Force Plot showing how features push prediction up or down")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

st.markdown("""
### 💡 Tips for Interpretation

**High SHAP Value (Red):** Feature strongly pushes prediction towards CKD
**Low SHAP Value (Blue):** Feature strongly pushes prediction towards Not CKD
**Near Zero:** Feature has little impact on this prediction

**Remember:** SHAP explains the model's decision, not necessarily medical causation!
""")

# Download option
if st.session_state.shap_computed:
    st.subheader("💾 Export SHAP Values")
    
    # Create export dataframe
    if isinstance(st.session_state.shap_values, list):
        shap_df = pd.DataFrame(
            st.session_state.shap_values[1],
            columns=[f"shap_{col}" for col in X_sample.columns]
        )
    else:
        shap_df = pd.DataFrame(
            st.session_state.shap_values,
            columns=[f"shap_{col}" for col in X_sample.columns]
        )
    
    # Combine with original data
    export_df = pd.concat([X_sample.reset_index(drop=True), shap_df], axis=1)
    
    csv = export_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download SHAP Values (CSV)",
        data=csv,
        file_name=f"shap_values_{model_name}.csv",
        mime="text/csv"
    )