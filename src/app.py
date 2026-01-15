"""
Streamlit Application for CKD Detection Project
Interactive dashboard for data exploration and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

from step01_data_loading import DataLoader
from step02_data_processing import DataProcessor

# Page configuration
st.set_page_config(
    page_title="CKD Detection Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'df' not in st.session_state:
        st.session_state.df = None


def load_data_section():
    """Data loading section"""
    st.header("📥 Data Loading")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        data_path = st.text_input(
            "Data Path", 
            value="../../data/raw/chronic_kindey_disease.csv",
            help="Path to the raw CSV file"
        )
    
    with col2:
        st.write("")
        st.write("")
        if st.button("Load Data", type="primary", use_container_width=True):
            with st.spinner("Loading data..."):
                try:
                    loader = DataLoader(data_path)
                    df = loader.load_data()
                    
                    if loader.validate_schema():
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
                    else:
                        st.error("❌ Schema validation failed!")
                        
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
    
    # Display loaded data info
    if st.session_state.data_loaded:
        st.divider()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", st.session_state.df.shape[0])
        with col2:
            st.metric("Total Features", st.session_state.df.shape[1])
        with col3:
            numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).shape[1]
            st.metric("Numeric Features", numeric_cols)
        with col4:
            categorical_cols = st.session_state.df.select_dtypes(include=['object']).shape[1]
            st.metric("Categorical Features", categorical_cols)
        
        # Show data preview
        with st.expander("📊 Data Preview", expanded=True):
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
        
        # Show data types
        with st.expander("📋 Data Types"):
            dtype_df = pd.DataFrame({
                'Column': st.session_state.df.dtypes.index,
                'Data Type': st.session_state.df.dtypes.values,
                'Non-Null Count': st.session_state.df.count().values,
                'Null Count': st.session_state.df.isnull().sum().values
            })
            st.dataframe(dtype_df, use_container_width=True)


def process_data_section():
    """Data processing section"""
    st.header("⚙️ Data Processing")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first!")
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("Process Data", type="primary", use_container_width=True):
            with st.spinner("Processing data..."):
                try:
                    # Create temporary processor
                    processor = DataProcessor()
                    processor.df = st.session_state.df.copy()
                    
                    # Clean data
                    processor.clean_data()
                    st.session_state.df = processor.df
                    st.session_state.data_processed = True
                    
                    st.success("✅ Data processed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing data: {str(e)}")
    
    if st.session_state.data_processed:
        st.divider()
        
        # Processing results
        st.subheader("Processing Results")
        
        # Target distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Target Variable Distribution**")
            target_counts = st.session_state.df['status'].value_counts()
            st.bar_chart(target_counts)
        
        with col2:
            st.write("**Target Statistics**")
            for status, count in target_counts.items():
                percentage = (count / len(st.session_state.df)) * 100
                st.metric(
                    label=f"{status.upper()}", 
                    value=count, 
                    delta=f"{percentage:.1f}%"
                )


def analysis_section():
    """Data analysis section"""
    st.header("📊 Data Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first!")
        return
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Missing Values", "Statistical Summary", "Distributions"])
    
    with tab1:
        st.subheader("Missing Values Analysis")
        
        missing_count = st.session_state.df.isnull().sum()
        missing_percent = (st.session_state.df.isnull().sum() / len(st.session_state.df)) * 100
        
        missing_df = pd.DataFrame({
            'Feature': missing_count.index,
            'Missing Count': missing_count.values,
            'Missing %': missing_percent.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(missing_df['Feature'], missing_df['Missing %'], color='coral')
                ax.set_xlabel('Missing Percentage (%)')
                ax.set_title('Missing Values by Feature')
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.dataframe(
                    missing_df.style.background_gradient(subset=['Missing %'], cmap='Reds'),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.success("✅ No missing values found!")
    
    with tab2:
        st.subheader("Statistical Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numerical Features**")
            numeric_summary = st.session_state.df.describe()
            st.dataframe(numeric_summary, use_container_width=True)
        
        with col2:
            st.write("**Categorical Features**")
            categorical_summary = st.session_state.df.describe(include=['object'])
            st.dataframe(categorical_summary, use_container_width=True)
    
    with tab3:
        st.subheader("Feature Distributions")
        
        # Feature selector
        numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_feature = st.selectbox("Select Feature", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Histogram**")
                fig, ax = plt.subplots(figsize=(8, 4))
                st.session_state.df[selected_feature].dropna().hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
                ax.set_xlabel(selected_feature)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {selected_feature}')
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.write("**Box Plot**")
                fig, ax = plt.subplots(figsize=(8, 4))
                st.session_state.df.boxplot(column=selected_feature, ax=ax)
                ax.set_ylabel(selected_feature)
                ax.set_title(f'Box Plot of {selected_feature}')
                st.pyplot(fig)
                plt.close()


def sidebar():
    """Sidebar configuration"""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/kidney.png", width=100)
        st.title("CKD Detection")
        st.markdown("---")
        
        st.subheader("Navigation")
        page = st.radio(
            "Go to",
            ["Data Loading", "Data Processing", "Data Analysis"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        st.subheader("Status")
        if st.session_state.data_loaded:
            st.success("✅ Data Loaded")
        else:
            st.info("⏳ Data Not Loaded")
        
        if st.session_state.data_processed:
            st.success("✅ Data Processed")
        else:
            st.info("⏳ Data Not Processed")
        
        st.markdown("---")
        
        st.subheader("About")
        st.info(
            "**CKD Detection Dashboard**\n\n"
            "Interactive tool for Chronic Kidney Disease detection analysis.\n\n"
            "Built with Streamlit 🎈"
        )
        
        return page


def main():
    """Main application"""
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Chronic Kidney Disease Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    page = sidebar()
    
    # Main content based on selected page
    st.markdown("---")
    
    if page == "Data Loading":
        load_data_section()
    elif page == "Data Processing":
        process_data_section()
    elif page == "Data Analysis":
        analysis_section()


if __name__ == "__main__":
    main()