import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🎯 Advanced Attendance Analytics Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for styling
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    
    /* Enhanced KPI Card Styling */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 2.5rem 2rem;
        border-radius: 20px;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.25);
        margin: 1.5rem 0;
        border: 3px solid rgba(255, 255, 255, 0.2);
        transform: translateY(0);
        transition: all 0.4s ease;
        backdrop-filter: blur(10px);
    }
    
    .stMetric:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.35);
        border-color: rgba(255, 255, 255, 0.4);
    }
    
    .stMetric [data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        padding: 2.5rem 2rem;
        border: 3px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.25);
    }
    
    .stMetric [data-testid="metric-container"] > div {
        color: white !important;
    }
    
    .stMetric [data-testid="metric-container"] [data-testid="metric-value"] {
        font-size: 3rem !important;
        font-weight: 900 !important;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.4);
        color: white !important;
    }
    
    .stMetric [data-testid="metric-container"] [data-testid="metric-label"] {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        opacity: 0.95;
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .stMetric [data-testid="metric-container"] [data-testid="metric-delta"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        opacity: 0.9;
        color: white !important;
    }
    
    .big-font {
        font-size: 36px !important;
        font-weight: 800;
        color: #2c3e50;
        margin-bottom: 2rem;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid #28a745;
        margin: 1.5rem 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .header-style {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    .filter-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 2px solid #dee2e6;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .outlier-table {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid #ffc107;
        margin: 1.5rem 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Enhanced spacing for KPI section */
    .kpi-section {
        padding: 3rem 0;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and process the attendance forecast data with enhanced preprocessing."""
    try:
        # Load the specific CSV file
        df = pd.read_csv('attendance_forecast_results_20250609_191409.csv')
        
        # Parse WEEK_BEGIN as datetime
        df['WEEK_BEGIN'] = pd.to_datetime(df['WEEK_BEGIN'])
        
        # Extract temporal features
        df['YEAR'] = df['WEEK_BEGIN'].dt.year
        df['MONTH'] = df['WEEK_BEGIN'].dt.month
        df['QUARTER'] = df['WEEK_BEGIN'].dt.quarter
        df['WEEK_NUMBER'] = df['WEEK_BEGIN'].dt.isocalendar().week
        df['MONTH_NAME'] = df['WEEK_BEGIN'].dt.month_name()
        df['QUARTER_NAME'] = 'Q' + df['QUARTER'].astype(str) + ' ' + df['YEAR'].astype(str)
        
        # Compute errors and metrics
        df['GREYKITE_ERROR'] = df['GREYKITE_FORECAST'] - df['ACTUAL_ATTENDANCE_RATE']
        df['MA_ERROR'] = df['MOVING_AVG_4WEEK_FORECAST'] - df['ACTUAL_ATTENDANCE_RATE']
        
        # Absolute errors
        df['GREYKITE_ABS_ERROR'] = np.abs(df['GREYKITE_ERROR'])
        df['MA_ABS_ERROR'] = np.abs(df['MA_ERROR'])
        
        # Percentage errors
        df['GREYKITE_APE'] = np.abs(df['GREYKITE_ERROR']) / np.abs(df['ACTUAL_ATTENDANCE_RATE']) * 100
        df['MA_APE'] = np.abs(df['MA_ERROR']) / np.abs(df['ACTUAL_ATTENDANCE_RATE']) * 100
        
        # Replace infinite values
        df['GREYKITE_APE'] = df['GREYKITE_APE'].replace([np.inf, -np.inf], np.nan)
        df['MA_APE'] = df['MA_APE'].replace([np.inf, -np.inf], np.nan)
        
        # Squared errors for MSE/RMSE
        df['GREYKITE_SE'] = df['GREYKITE_ERROR'] ** 2
        df['MA_SE'] = df['MA_ERROR'] ** 2
        
        # Performance indicators
        df['GREYKITE_WINS'] = (df['GREYKITE_ABS_ERROR'] < df['MA_ABS_ERROR']).astype(int)
        df['PERFORMANCE_IMPROVEMENT'] = ((df['MA_ABS_ERROR'] - df['GREYKITE_ABS_ERROR']) / df['MA_ABS_ERROR']) * 100
        
        # Outlier detection (using IQR method)
        for col in ['GREYKITE_APE', 'MA_APE']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[f'{col}_OUTLIER'] = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).astype(int)
        
        return df
        
    except FileNotFoundError:
        st.error("⚠️ File 'attendance_forecast_results_20250609_191409.csv' not found!")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading data: {str(e)}")
        return None

def create_header():
    """Create enhanced dashboard header"""
    st.markdown("""
    <div class="header-style">
        <h1 style="font-size: 4rem; margin: 0; text-shadow: 3px 3px 6px rgba(0,0,0,0.4);">
            ⚡ ADVANCED ATTENDANCE ANALYTICS
        </h1>
        <h2 style="font-size: 1.8rem; margin: 1rem 0 0 0; opacity: 0.95;">
            🎯 Comprehensive Forecast Performance Dashboard
        </h2>
        <p style="font-size: 1.2rem; margin: 1rem 0 0 0; opacity: 0.85;">
            📊 Real-time Analytics • 🔍 Interactive Filtering • 📈 Advanced Insights
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_filters(df):
    """Create comprehensive sidebar filters."""
    st.sidebar.markdown("### 🎛️ DASHBOARD FILTERS")
    
    filters = {}
    
    # Year filter
    available_years = sorted(df['YEAR'].unique())
    filters['years'] = st.sidebar.multiselect(
        "📅 Select Years",
        options=available_years,
        default=available_years,
        help="Choose which years to include in analysis"
    )
    
    # Date range filter
    min_date = df['WEEK_BEGIN'].min().date()
    max_date = df['WEEK_BEGIN'].max().date()
    
    date_range = st.sidebar.date_input(
        "📆 Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Select custom date range for analysis"
    )
    
    if len(date_range) == 2:
        filters['start_date'] = pd.to_datetime(date_range[0])
        filters['end_date'] = pd.to_datetime(date_range[1])
    else:
        filters['start_date'] = pd.to_datetime(min_date)
        filters['end_date'] = pd.to_datetime(max_date)
    
    # Quarter filter
    available_quarters = sorted(df['QUARTER_NAME'].unique())
    filters['quarters'] = st.sidebar.multiselect(
        "📊 Select Quarters",
        options=available_quarters,
        default=available_quarters,
        help="Filter by specific quarters"
    )
    
    # Location filter
    available_locations = sorted(df['WORK_LOCATION'].unique())
    filters['locations'] = st.sidebar.multiselect(
        "🏢 Work Locations",
        options=available_locations,
        default=available_locations,
        help="Select specific work locations"
    )
    
    # Department filter
    available_departments = sorted(df['DEPARTMENT_GROUP'].unique())
    filters['departments'] = st.sidebar.multiselect(
        "🏭 Department Groups",
        options=available_departments,
        default=available_departments,
        help="Choose department groups to analyze"
    )
    
    # Shift filter
    available_shifts = sorted(df['SHIFT_TIME'].unique())
    filters['shifts'] = st.sidebar.multiselect(
        "⏰ Shift Times",
        options=available_shifts,
        default=available_shifts,
        help="Filter by specific shift times"
    )
    
    # Performance filter
    filters['performance'] = st.sidebar.selectbox(
        "🎯 Performance Filter",
        options=["All Data", "Greykite Wins Only", "MA Wins Only", "Close Performance (±5%)"],
        help="Filter data based on model performance"
    )
    
    # Reset filters button
    if st.sidebar.button("🔄 Reset All Filters", help="Reset all filters to default values"):
        st.experimental_rerun()
    
    return filters

def apply_filters(df, filters):
    """Apply selected filters to the dataframe."""
    filtered_df = df.copy()
    
    # Apply year filter
    if filters['years']:
        filtered_df = filtered_df[filtered_df['YEAR'].isin(filters['years'])]
    
    # Apply date range filter
    filtered_df = filtered_df[
        (filtered_df['WEEK_BEGIN'] >= filters['start_date']) & 
        (filtered_df['WEEK_BEGIN'] <= filters['end_date'])
    ]
    
    # Apply quarter filter
    if filters['quarters']:
        filtered_df = filtered_df[filtered_df['QUARTER_NAME'].isin(filters['quarters'])]
    
    # Apply location filter
    if filters['locations']:
        filtered_df = filtered_df[filtered_df['WORK_LOCATION'].isin(filters['locations'])]
    
    # Apply department filter
    if filters['departments']:
        filtered_df = filtered_df[filtered_df['DEPARTMENT_GROUP'].isin(filters['departments'])]
    
    # Apply shift filter
    if filters['shifts']:
        filtered_df = filtered_df[filtered_df['SHIFT_TIME'].isin(filters['shifts'])]
    
    # Apply performance filter
    if filters['performance'] == "Greykite Wins Only":
        filtered_df = filtered_df[filtered_df['GREYKITE_WINS'] == 1]
    elif filters['performance'] == "MA Wins Only":
        filtered_df = filtered_df[filtered_df['GREYKITE_WINS'] == 0]
    elif filters['performance'] == "Close Performance (±5%)":
        filtered_df = filtered_df[abs(filtered_df['PERFORMANCE_IMPROVEMENT']) <= 5]
    
    return filtered_df

def show_data_info(df, filtered_df):
    """Display enhanced data information in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 DATASET OVERVIEW")
    
    # Original data info
    st.sidebar.markdown("**📈 Original Dataset:**")
    st.sidebar.markdown(f"""
    • **Total Records**: {len(df):,}  
    • **Locations**: {df['WORK_LOCATION'].nunique()}  
    • **Departments**: {df['DEPARTMENT_GROUP'].nunique()}  
    • **Shifts**: {df['SHIFT_TIME'].nunique()}  
    • **Date Range**: {(df['WEEK_BEGIN'].max() - df['WEEK_BEGIN'].min()).days} days  
    """)
    
    # Filtered data info
    st.sidebar.markdown("**🔍 Filtered Dataset:**")
    filter_percentage = (len(filtered_df) / len(df)) * 100 if len(df) > 0 else 0
    st.sidebar.markdown(f"""
    • **Filtered Records**: {len(filtered_df):,} ({filter_percentage:.1f}%)  
    • **Date Coverage**: {filtered_df['WEEK_BEGIN'].min().strftime('%Y-%m-%d')} to {filtered_df['WEEK_BEGIN'].max().strftime('%Y-%m-%d')}  
    • **Avg Attendance**: {filtered_df['ACTUAL_ATTENDANCE_RATE'].mean():.1f}%  
    • **Greykite Wins**: {filtered_df['GREYKITE_WINS'].sum()}/{len(filtered_df)} ({filtered_df['GREYKITE_WINS'].mean()*100:.1f}%)  
    """)

def create_executive_summary(df):
    """Create clean executive summary with key insights only."""
    st.markdown('<p class="big-font">🎯 EXECUTIVE PERFORMANCE SUMMARY</p>', unsafe_allow_html=True)
    
    if len(df) == 0:
        st.warning("⚠️ No data available for the selected filters.")
        return
    
    # Calculate key metrics
    total_weeks = len(df)
    greykite_wins = df['GREYKITE_WINS'].sum()
    win_rate = (greykite_wins / total_weeks) * 100 if total_weeks > 0 else 0
    
    avg_greykite_mape = df['GREYKITE_APE'].mean()
    avg_ma_mape = df['MA_APE'].mean()
    mape_improvement = ((avg_ma_mape - avg_greykite_mape) / avg_ma_mape) * 100 if avg_ma_mape != 0 else 0
    
    # Performance by segment
    if len(df) > 0:
        best_location = df.groupby('WORK_LOCATION')['GREYKITE_WINS'].mean().idxmax()
        best_department = df.groupby('DEPARTMENT_GROUP')['GREYKITE_WINS'].mean().idxmax()
    else:
        best_location = "N/A"
        best_department = "N/A"
    
    # Trend analysis
    recent_data = df.tail(min(12, len(df)))  # Last 12 weeks or available data
    recent_win_rate = recent_data['GREYKITE_WINS'].mean() * 100 if len(recent_data) > 0 else 0
    trend = "📈 IMPROVING" if recent_win_rate > win_rate else "📉 DECLINING" if recent_win_rate < win_rate - 5 else "➡️ STABLE"
    
    st.markdown(f"""
    <div class="success-box">
    <h3>🏆 KEY PERFORMANCE HIGHLIGHTS</h3>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 3rem; font-size: 1.2rem;">
        <div>
            <ul style="list-style-type: none; padding: 0;">
                <li style="margin: 1rem 0;"><strong>📊 Overall Win Rate:</strong> {win_rate:.1f}% ({greykite_wins}/{total_weeks} weeks)</li>
                <li style="margin: 1rem 0;"><strong>📈 MAPE Improvement:</strong> {mape_improvement:.1f}% better than Moving Average</li>
            </ul>
        </div>
        <div>
            <ul style="list-style-type: none; padding: 0;">
                <li style="margin: 1rem 0;"><strong>🔄 Recent Trend:</strong> {trend}</li>
                <li style="margin: 1rem 0;"><strong>🏢 Best Location:</strong> {best_location}</li>
                <li style="margin: 1rem 0;"><strong>🏭 Best Department:</strong> {best_department}</li>
            </ul>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

def create_enhanced_kpi_metrics(df):
    """Create enhanced 3-column KPI metrics dashboard with superior styling."""
    st.markdown("### 📊 KEY PERFORMANCE INDICATORS")
    st.markdown('<div class="kpi-section">', unsafe_allow_html=True)
    
    if len(df) == 0:
        st.warning("⚠️ No data available for the selected filters.")
        return
    
    # Calculate core metrics
    total_weeks = len(df)
    greykite_wins = df['GREYKITE_WINS'].sum()
    win_rate = (greykite_wins / total_weeks) * 100 if total_weeks > 0 else 0
    
    # MAPE metrics
    avg_greykite_mape = df['GREYKITE_APE'].mean()
    avg_ma_mape = df['MA_APE'].mean()
    mape_improvement = ((avg_ma_mape - avg_greykite_mape) / avg_ma_mape) * 100 if avg_ma_mape != 0 else 0
    
    # Create enhanced 3-column metrics display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🏆 Greykite Win Rate",
            f"{win_rate:.1f}%",
            delta=f"{greykite_wins}/{total_weeks} weeks",
            help="Percentage of weeks where Greykite outperformed Moving Average forecasts"
        )
    
    with col2:
        st.metric(
            "📈 MAPE Improvement",
            f"{mape_improvement:.2f}%",
            delta="vs Moving Average",
            help="Mean Absolute Percentage Error improvement over Moving Average baseline"
        )
    
    with col3:
        st.metric(
            "🎯 Greykite MAPE",
            f"{avg_greykite_mape:.2f}%",
            delta=f"MA: {avg_ma_mape:.2f}%",
            help="Average Mean Absolute Percentage Error for Greykite forecasting model"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_outliers_table(df):
    """Create comprehensive outliers analysis with MAPE focus."""
    st.markdown("### 🚨 OUTLIERS ANALYSIS")
    
    if len(df) == 0:
        st.warning("⚠️ No data available for the selected filters.")
        return
    
    # Get outliers based on both Greykite and MA APE
    outliers_df = df[
        (df['GREYKITE_APE_OUTLIER'] == 1) | (df['MA_APE_OUTLIER'] == 1)
    ].copy()
    
    if len(outliers_df) == 0:
        st.info("✅ No outliers detected in the current filtered dataset.")
    else:
        # Display summary metrics with MAPE focus
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🚨 Total Outliers", len(outliers_df))
        with col2:
            greykite_outliers = len(outliers_df[outliers_df['GREYKITE_APE_OUTLIER'] == 1])
            st.metric("🎯 Greykite Outliers", greykite_outliers)
        with col3:
            ma_outliers = len(outliers_df[outliers_df['MA_APE_OUTLIER'] == 1])
            st.metric("📊 MA Outliers", ma_outliers)
    
    # Calculate MAPE by segments for outlier analysis
    st.markdown("""
    <div class="outlier-table">
    <h4>📋 MAPE Analysis by Segments (Outlier Impact)</h4>
    <p><strong>Note:</strong> Outliers are detected using the IQR method, showing MAPE impact by different segments</p>
    </div>
    """, unsafe_allow_html=True)
    
    # MAPE by Location
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏢 MAPE by Location:**")
        location_mape = df.groupby('WORK_LOCATION').agg({
            'GREYKITE_APE': 'mean',
            'MA_APE': 'mean',
            'GREYKITE_APE_OUTLIER': 'sum',
            'MA_APE_OUTLIER': 'sum'
        }).round(2)
        location_mape.columns = ['Greykite MAPE (%)', 'MA MAPE (%)', 'Greykite Outliers', 'MA Outliers']
        st.dataframe(location_mape, use_container_width=True)
    
    with col2:
        st.markdown("**🏭 MAPE by Department:**")
        dept_mape = df.groupby('DEPARTMENT_GROUP').agg({
            'GREYKITE_APE': 'mean',
            'MA_APE': 'mean',
            'GREYKITE_APE_OUTLIER': 'sum',
            'MA_APE_OUTLIER': 'sum'
        }).round(2)
        dept_mape.columns = ['Greykite MAPE (%)', 'MA MAPE (%)', 'Greykite Outliers', 'MA Outliers']
        st.dataframe(dept_mape, use_container_width=True)
    
    # Overall MAPE comparison
    overall_greykite_mape = df['GREYKITE_APE'].mean()
    overall_ma_mape = df['MA_APE'].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **🎯 Overall Performance:**
        - **Greykite MAPE**: {overall_greykite_mape:.2f}%
        - **Total Greykite Outliers**: {df['GREYKITE_APE_OUTLIER'].sum()}
        """)
    
    with col2:
        st.markdown(f"""
        **📊 Baseline Performance:**
        - **MA MAPE**: {overall_ma_mape:.2f}%
        - **Total MA Outliers**: {df['MA_APE_OUTLIER'].sum()}
        """)

def create_performance_timeline(df):
    """Create comprehensive performance timeline."""
    st.markdown("### 📅 PERFORMANCE TIMELINE ANALYSIS")
    
    if len(df) == 0:
        st.warning("⚠️ No data available for the selected filters.")
        return
    
    # Prepare data for timeline
    timeline_data = df.groupby('WEEK_BEGIN').agg({
        'GREYKITE_APE': ['mean', 'std'],
        'MA_APE': ['mean', 'std'],
        'GREYKITE_WINS': 'mean',
        'PERFORMANCE_IMPROVEMENT': 'mean'
    }).reset_index()
    
    timeline_data.columns = [
        'WEEK_BEGIN', 'GREYKITE_MAPE_MEAN', 'GREYKITE_MAPE_STD',
        'MA_MAPE_MEAN', 'MA_MAPE_STD', 'WIN_RATE', 'IMPROVEMENT'
    ]
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "📈 MAPE Comparison Over Time", 
            "🏆 Weekly Win Rate", 
            "📊 Performance Improvement", 
            "📉 Error Trends"
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # MAPE comparison
    fig.add_trace(
        go.Scatter(
            x=timeline_data['WEEK_BEGIN'],
            y=timeline_data['GREYKITE_MAPE_MEAN'],
            name='Greykite MAPE',
            line=dict(color='#1f77b4', width=3),
            mode='lines+markers'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=timeline_data['WEEK_BEGIN'],
            y=timeline_data['MA_MAPE_MEAN'],
            name='Moving Avg MAPE',
            line=dict(color='#ff7f0e', width=3),
            mode='lines+markers'
        ),
        row=1, col=1
    )
    
    # Win rate
    fig.add_trace(
        go.Scatter(
            x=timeline_data['WEEK_BEGIN'],
            y=timeline_data['WIN_RATE'] * 100,
            name='Win Rate %',
            line=dict(color='#2ca02c', width=3),
            mode='lines+markers',
            fill='tonexty'
        ),
        row=1, col=2
    )
    
    # Performance improvement
    fig.add_trace(
        go.Scatter(
            x=timeline_data['WEEK_BEGIN'],
            y=timeline_data['IMPROVEMENT'],
            name='Improvement %',
            line=dict(color='#d62728', width=3),
            mode='lines+markers'
        ),
        row=2, col=1
    )
    
    # Error trends
    fig.add_trace(
        go.Scatter(
            x=timeline_data['WEEK_BEGIN'],
            y=timeline_data['GREYKITE_MAPE_STD'],
            name='Greykite MAPE Std',
            line=dict(color='#9467bd', width=2),
            mode='lines'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=timeline_data['WEEK_BEGIN'],
            y=timeline_data['MA_MAPE_STD'],
            name='MA MAPE Std',
            line=dict(color='#8c564b', width=2),
            mode='lines'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text="📊 COMPREHENSIVE PERFORMANCE TIMELINE",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_segment_analysis(df):
    """Create comprehensive MAPE-focused segment analysis."""
    st.markdown("### 🎯 MAPE PERFORMANCE BY SEGMENTS")
    
    if len(df) == 0:
        st.warning("⚠️ No data available for the selected filters.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Location MAPE performance
        if df['WORK_LOCATION'].nunique() > 1:
            location_performance = df.groupby('WORK_LOCATION').agg({
                'GREYKITE_WINS': 'mean',
                'GREYKITE_APE': 'mean',  # This will be displayed as MAPE
                'MA_APE': 'mean',        # This will be displayed as MAPE
                'PERFORMANCE_IMPROVEMENT': 'mean'
            }).round(2).reset_index()
            
            location_performance['WIN_RATE'] = (location_performance['GREYKITE_WINS'] * 100).round(1)
            location_performance['MAPE_IMPROVEMENT'] = (
                ((location_performance['MA_APE'] - location_performance['GREYKITE_APE']) / 
                 location_performance['MA_APE']) * 100
            ).round(2)
            
            # Create MAPE comparison chart
            fig_loc = px.bar(
                location_performance.sort_values('GREYKITE_APE', ascending=True),
                x='GREYKITE_APE',
                y='WORK_LOCATION',
                title="🏢 GREYKITE MAPE BY LOCATION",
                labels={'GREYKITE_APE': 'Greykite MAPE (%)', 'WORK_LOCATION': 'Location'},
                color='MAPE_IMPROVEMENT',
                color_continuous_scale='RdYlGn',
                text='WIN_RATE'
            )
            fig_loc.update_traces(texttemplate='%{text}% Win Rate', textposition='outside')
            fig_loc.update_layout(height=500)
            st.plotly_chart(fig_loc, use_container_width=True)
            
            # Display MAPE summary table
            st.markdown("**📊 Location MAPE Summary:**")
            summary_display = location_performance[['WORK_LOCATION', 'GREYKITE_APE', 'MA_APE', 'WIN_RATE', 'MAPE_IMPROVEMENT']].copy()
            summary_display.columns = ['Location', 'Greykite MAPE (%)', 'MA MAPE (%)', 'Win Rate (%)', 'MAPE Improvement (%)']
            st.dataframe(summary_display, use_container_width=True)
        else:
            st.info("Only one location in filtered data")
    
    with col2:
        # Department MAPE performance
        if df['DEPARTMENT_GROUP'].nunique() > 1:
            dept_performance = df.groupby('DEPARTMENT_GROUP').agg({
                'GREYKITE_WINS': 'mean',
                'GREYKITE_APE': 'mean',  # This will be displayed as MAPE
                'MA_APE': 'mean',        # This will be displayed as MAPE
                'PERFORMANCE_IMPROVEMENT': 'mean'
            }).round(2).reset_index()
            
            dept_performance['WIN_RATE'] = (dept_performance['GREYKITE_WINS'] * 100).round(1)
            dept_performance['MAPE_IMPROVEMENT'] = (
                ((dept_performance['MA_APE'] - dept_performance['GREYKITE_APE']) / 
                 dept_performance['MA_APE']) * 100
            ).round(2)
            
            # Create MAPE comparison chart
            fig_dept = px.bar(
                dept_performance.sort_values('GREYKITE_APE', ascending=True),
                x='GREYKITE_APE',
                y='DEPARTMENT_GROUP',
                title="🏭 GREYKITE MAPE BY DEPARTMENT",
                labels={'GREYKITE_APE': 'Greykite MAPE (%)', 'DEPARTMENT_GROUP': 'Department'},
                color='MAPE_IMPROVEMENT',
                color_continuous_scale='RdYlGn',
                text='WIN_RATE'
            )
            fig_dept.update_traces(texttemplate='%{text}% Win Rate', textposition='outside')
            fig_dept.update_layout(height=500)
            st.plotly_chart(fig_dept, use_container_width=True)
            
            # Display MAPE summary table
            st.markdown("**📊 Department MAPE Summary:**")
            summary_display = dept_performance[['DEPARTMENT_GROUP', 'GREYKITE_APE', 'MA_APE', 'WIN_RATE', 'MAPE_IMPROVEMENT']].copy()
            summary_display.columns = ['Department', 'Greykite MAPE (%)', 'MA MAPE (%)', 'Win Rate (%)', 'MAPE Improvement (%)']
            st.dataframe(summary_display, use_container_width=True)
        else:
            st.info("Only one department in filtered data")

def create_error_analysis(df):
    """Create MAPE-focused analysis and summary."""
    st.markdown("### 📊 MAPE ANALYSIS & SUMMARY")
    
    if len(df) == 0:
        st.warning("⚠️ No data available for the selected filters.")
        return
    
    # Calculate overall MAPE statistics
    greykite_mape = df['GREYKITE_APE'].mean()
    ma_mape = df['MA_APE'].mean()
    mape_improvement = ((ma_mape - greykite_mape) / ma_mape) * 100 if ma_mape != 0 else 0
    
    # Display overall MAPE metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🎯 Greykite MAPE",
            f"{greykite_mape:.2f}%",
            help="Mean Absolute Percentage Error for Greykite model"
        )
    
    with col2:
        st.metric(
            "📊 Moving Average MAPE",
            f"{ma_mape:.2f}%",
            help="Mean Absolute Percentage Error for Moving Average baseline"
        )
    
    with col3:
        st.metric(
            "📈 MAPE Improvement",
            f"{mape_improvement:.2f}%",
            delta="vs MA baseline",
            help="Percentage improvement in MAPE compared to Moving Average"
        )
    
    st.markdown("---")
    
    # MAPE comparison visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # MAPE comparison bar chart
        mape_comparison = pd.DataFrame({
            'Model': ['Greykite', 'Moving Average'],
            'MAPE': [greykite_mape, ma_mape],
            'Color': ['#1f77b4', '#ff7f0e']
        })
        
        fig_bar = px.bar(
            mape_comparison,
            x='Model',
            y='MAPE',
            title="📊 MAPE Comparison",
            color='Model',
            text='MAPE'
        )
        fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_bar.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # MAPE statistics table
        st.markdown("**📋 MAPE Statistical Summary:**")
        
        mape_stats = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th Percentile', '75th Percentile'],
            'Greykite MAPE (%)': [
                df['GREYKITE_APE'].mean(),
                df['GREYKITE_APE'].median(),
                df['GREYKITE_APE'].std(),
                df['GREYKITE_APE'].min(),
                df['GREYKITE_APE'].max(),
                df['GREYKITE_APE'].quantile(0.25),
                df['GREYKITE_APE'].quantile(0.75)
            ],
            'MA MAPE (%)': [
                df['MA_APE'].mean(),
                df['MA_APE'].median(),
                df['MA_APE'].std(),
                df['MA_APE'].min(),
                df['MA_APE'].max(),
                df['MA_APE'].quantile(0.25),
                df['MA_APE'].quantile(0.75)
            ]
        }).round(2)
        
        st.dataframe(mape_stats, use_container_width=True)
    
    # Performance insights
    st.markdown("---")
    st.markdown("### 🎯 MAPE PERFORMANCE INSIGHTS")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Best and worst MAPE periods
        best_week_idx = df['GREYKITE_APE'].idxmin()
        worst_week_idx = df['GREYKITE_APE'].idxmax()
        
        best_week = df.loc[best_week_idx]
        worst_week = df.loc[worst_week_idx]
        
        st.markdown(f"""
        **🏆 Best Greykite Performance:**
        - **Week**: {best_week['WEEK_BEGIN'].strftime('%Y-%m-%d')}
        - **MAPE**: {best_week['GREYKITE_APE']:.2f}%
        - **Location**: {best_week['WORK_LOCATION']}
        - **Department**: {best_week['DEPARTMENT_GROUP']}
        """)
        
        st.markdown(f"""
        **⚠️ Worst Greykite Performance:**
        - **Week**: {worst_week['WEEK_BEGIN'].strftime('%Y-%m-%d')}
        - **MAPE**: {worst_week['GREYKITE_APE']:.2f}%
        - **Location**: {worst_week['WORK_LOCATION']}
        - **Department**: {worst_week['DEPARTMENT_GROUP']}
        """)
    
    with col2:
        # Model consistency analysis
        greykite_consistency = df['GREYKITE_APE'].std()
        ma_consistency = df['MA_APE'].std()
        
        win_rate = (df['GREYKITE_WINS'].sum() / len(df)) * 100
        
        st.markdown(f"""
        **📊 Model Consistency (Lower is Better):**
        - **Greykite MAPE Std Dev**: {greykite_consistency:.2f}%
        - **MA MAPE Std Dev**: {ma_consistency:.2f}%
        - **Consistency Advantage**: {'Greykite' if greykite_consistency < ma_consistency else 'Moving Average'}
        
        **🏆 Overall Performance:**
        - **Win Rate**: {win_rate:.1f}%
        - **Total Predictions**: {len(df):,}
        - **Greykite Wins**: {df['GREYKITE_WINS'].sum():,}
        """)

def main():
    """Main dashboard application."""
    # Create enhanced header
    create_header()
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Create filters
    filters = create_filters(df)
    
    # Apply filters
    filtered_df = apply_filters(df, filters)
    
    # Show data info in sidebar
    show_data_info(df, filtered_df)
    
    # Display filter summary
    if len(filtered_df) != len(df):
        filter_summary = []
        if len(filters['years']) < len(df['YEAR'].unique()):
            filter_summary.append(f"Years: {', '.join(map(str, filters['years']))}")
        if len(filters['locations']) < len(df['WORK_LOCATION'].unique()):
            filter_summary.append(f"Locations: {len(filters['locations'])} selected")
        if len(filters['departments']) < len(df['DEPARTMENT_GROUP'].unique()):
            filter_summary.append(f"Departments: {len(filters['departments'])} selected")
        if len(filters['shifts']) < len(df['SHIFT_TIME'].unique()):
            filter_summary.append(f"Shifts: {len(filters['shifts'])} selected")
        if filters['performance'] != "All Data":
            filter_summary.append(f"Performance: {filters['performance']}")
        
        if filter_summary:
            st.info(f"🔍 **Active Filters:** {' | '.join(filter_summary)}")
    
    # Main dashboard sections
    create_executive_summary(filtered_df)
    st.markdown("---")
    
    create_enhanced_kpi_metrics(filtered_df)
    st.markdown("---")
    
    create_outliers_table(filtered_df)
    st.markdown("---")
    
    create_performance_timeline(filtered_df)
    st.markdown("---")
    
    create_segment_analysis(filtered_df)
    st.markdown("---")
    
    create_error_analysis(filtered_df)
    
    # Enhanced Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0; color: #7f8c8d;">
        <p style="font-size: 1.4rem; font-weight: 700;">⚡ ADVANCED ATTENDANCE ANALYTICS DASHBOARD</p>
        <p style="font-size: 1.1rem;">📊 Built with Streamlit & Plotly | 🔄 Real-time Updates | 📈 Interactive Analytics | 🎯 Enhanced User Experience</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
