# Streamlit Application Setup Guide

## 📁 Project Structure

```
ckd-detection/
│
├── data/
│   ├── raw/
│   │   └── chronic_kindey_disease.csv
│   └── processed/
│
├── src/
│   ├── __init__.py
│   ├── app.py                    # 🆕 Streamlit application
│   ├── data_loading.py           # Data loading module
│   └── data_processing.py        # Data processing module
│
├── figures/
├── reports/
│
├── requirements.txt              # Updated with Streamlit
└── README.md
```

## 🚀 Quick Start

### 1. Update Dependencies

Add Streamlit to your `requirements.txt`:

```bash
# Add to existing requirements.txt
streamlit==1.29.0
plotly==5.18.0  # Optional: for interactive plots
```

Or install directly:

```bash
pip install streamlit==1.29.0
```

### 2. Run the Application

```bash
# Navigate to src directory
cd src

# Run Streamlit app
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`

## 🎯 Application Features

### Page 1: Data Loading
- **Load CSV data** with custom path
- **Validate schema** automatically
- **View metrics**: Total records, features, data types
- **Preview data** in interactive table
- **Inspect data types** with null counts

### Page 2: Data Processing
- **Clean data**: Remove whitespace, standardize values
- **View processing results**: Target distribution
- **Statistics**: Class balance and percentages

### Page 3: Data Analysis
- **Missing Values**: Visual analysis with charts
- **Statistical Summary**: Numeric and categorical summaries
- **Distributions**: Interactive feature exploration
  - Histograms
  - Box plots
  - Feature selection

## 🎨 Key Components

### Session State Management
```python
st.session_state.data_loaded      # Boolean: Is data loaded?
st.session_state.data_processed   # Boolean: Is data processed?
st.session_state.df               # DataFrame: Current dataset
```

### Layout Structure
```
Sidebar                Main Content
├── Navigation    →    ├── Data Loading Section
├── Status             ├── Data Processing Section
└── About              └── Data Analysis Section
```

## 💡 Usage Flow

1. **Start the app** → Opens with Data Loading page
2. **Load Data** → Enter path and click "Load Data"
3. **View Loaded Data** → Check metrics and preview
4. **Process Data** → Navigate to Data Processing, click "Process Data"
5. **Analyze** → Navigate to Data Analysis, explore visualizations

## 🔧 Customization Options

### Modify Colors
Edit the CSS in `app.py`:
```python
st.markdown("""
    <style>
    .main-header {
        color: #YOUR_COLOR;  # Change header color
    }
    </style>
""", unsafe_allow_html=True)
```

### Add New Pages
```python
# In sidebar()
page = st.radio(
    "Go to",
    ["Data Loading", "Data Processing", "Data Analysis", "YOUR_NEW_PAGE"]
)

# In main()
elif page == "YOUR_NEW_PAGE":
    your_new_function()
```

### Change Default Path
```python
data_path = st.text_input(
    "Data Path", 
    value="YOUR_DEFAULT_PATH"  # Change this
)
```

## 🐛 Common Issues

### Issue 1: Module Import Error
```
ModuleNotFoundError: No module named 'data_loading'
```
**Solution**: Run from `src/` directory or adjust `sys.path`

### Issue 2: Data File Not Found
```
FileNotFoundError: [Errno 2] No such file or directory
```
**Solution**: Check the relative path from `src/` directory

### Issue 3: Port Already in Use
```
Port 8501 is in use
```
**Solution**: Use different port
```bash
streamlit run app.py --server.port 8502
```

## 📊 Advanced Features (Optional)

### Add Plotly Interactive Charts
```python
import plotly.express as px

fig = px.scatter(df, x='age', y='bp', color='status')
st.plotly_chart(fig)
```

### Add Download Buttons
```python
csv = df.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name='processed_data.csv',
    mime='text/csv'
)
```

### Add File Upload
```python
uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
```

## 🎯 Next Steps

1. **Add Model Training Page**: Create ML model training interface
2. **Add Prediction Page**: Upload new data for predictions
3. **Add Model Comparison**: Compare multiple models
4. **Add Export Features**: Download reports and figures

## 📝 Configuration File (Optional)

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
enableCORS = false
```

## 🔗 Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Streamlit Cheat Sheet](https://docs.streamlit.io/library/cheatsheet)