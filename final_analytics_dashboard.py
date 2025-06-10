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
        
        # Handle WEEK_NUMBER from CSV - keep original if exists, otherwise leave blank
        if 'WEEK_NUMBER' not in df.columns:
            df['WEEK_NUMBER'] = ''  # Keep blank if not in CSV
        else:
            # Handle missing values in WEEK_NUMBER - convert NaN to empty string
            df['WEEK_NUMBER'] = df['WEEK_NUMBER'].fillna('').astype(str)
        
        # Create a flag for missing week numbers
        df['WEEK_NUMBER_MISSING'] = (df['WEEK_NUMBER'] == '') | (df['WEEK_NUMBER'].isna())
        
        df['MONTH_NAME'] = df['WEEK_BEGIN'].dt.month_name()
        df['QUARTER_NAME'] = 'Q' + df['QUARTER'].astype(str) + ' ' + df['YEAR'].astype(str)
        
        # Compute errors and metrics
        df['GREYKITE_ERROR'] = df['GREYKITE_FORECAST'] - df['ACTUAL_ATTENDANCE_RATE']
        df['MA_ERROR'] = df['MOVING_AVG_4WEEK_FORECAST'] - df['ACTUAL_ATTENDANCE_RATE']
        
        # Absolute errors
        df['GREYKITE_ABS_ERROR'] = np.abs(df['GREYKITE_ERROR'])
        df['MA_ABS_ERROR'] = np.abs(df['MA_ERROR'])
        
        # Percentage errors (APE values for MAPE calculation)
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
    
    # Handle week coverage safely
    valid_weeks = filtered_df[filtered_df['WEEK_NUMBER'] != '']['WEEK_NUMBER']
    week_coverage = "N/A"
    if len(valid_weeks) > 0:
        week_coverage = f"{valid_weeks.min()} to {valid_weeks.max()}"
    
    # Check for missing week numbers
    missing_weeks = filtered_df['WEEK_NUMBER_MISSING'].sum()
    missing_week_info = f" (⚠️ {missing_weeks} missing)" if missing_weeks > 0 else ""
    
    st.sidebar.markdown(f"""
    • **Filtered Records**: {len(filtered_df):,} ({filter_percentage:.1f}%)  
    • **Week Coverage**: {week_coverage}{missing_week_info}  
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
    
    # Check for missing week numbers
    missing_weeks_count = df['WEEK_NUMBER_MISSING'].sum()
    data_quality_note = ""
    if missing_weeks_count > 0:
        data_quality_note = f"""
        <div style="background: #fff3cd; padding: 1rem; border-radius: 10px; border-left: 4px solid #ffc107; margin: 1rem 0;">
            <strong>⚠️ Data Quality Note:</strong> {missing_weeks_count} records have missing WEEK_NUMBER values and will show as blank in charts.
        </div>
        """
    
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
    {data_quality_note}
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
    """Create comprehensive outliers analysis table."""
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
        return
    
    # Prepare outliers table with requested columns
    outliers_display = outliers_df[[
        'WEEK_NUMBER', 'WORK_LOCATION', 'DEPARTMENT_GROUP', 'SHIFT_TIME',
        'ACTUAL_ATTENDANCE_RATE', 'GREYKITE_FORECAST', 'MOVING_AVG_4WEEK_FORECAST',
        'GREYKITE_APE', 'MA_APE', 'GREYKITE_APE_OUTLIER', 'MA_APE_OUTLIER'
    ]].copy()
    
    # Format the data for better display (keep WEEK_NUMBER as-is)
    outliers_display['ACTUAL_ATTENDANCE_RATE'] = outliers_display['ACTUAL_ATTENDANCE_RATE'].round(2)
    outliers_display['GREYKITE_FORECAST'] = outliers_display['GREYKITE_FORECAST'].round(2)
    outliers_display['MOVING_AVG_4WEEK_FORECAST'] = outliers_display['MOVING_AVG_4WEEK_FORECAST'].round(2)
    outliers_display['GREYKITE_APE'] = outliers_display['GREYKITE_APE'].round(2)
    outliers_display['MA_APE'] = outliers_display['MA_APE'].round(2)
    
    # Rename columns for better readability (displaying as MAPE values)
    outliers_display.columns = [
        'Week', 'Location', 'Department', 'Shift',
        'Actual Attendance (%)', 'Greykite Forecast (%)', '4-Week MA Forecast (%)',
        'Greykite MAPE (%)', 'MA MAPE (%)', 'Greykite Outlier', 'MA Outlier'
    ]
    
    # Create outlier type indicator
    outliers_display['Outlier Type'] = outliers_display.apply(
        lambda row: 'Both Models' if (row['Greykite Outlier'] == 1 and row['MA Outlier'] == 1)
        else 'Greykite Only' if row['Greykite Outlier'] == 1
        else 'MA Only', axis=1
    )
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🚨 Total Outliers", len(outliers_display))
    with col2:
        greykite_outliers = len(outliers_display[outliers_display['Greykite Outlier'] == 1])
        st.metric("🎯 Greykite Outliers", greykite_outliers)
    with col3:
        ma_outliers = len(outliers_display[outliers_display['MA Outlier'] == 1])
        st.metric("📊 MA Outliers", ma_outliers)
    
    st.markdown("""
    <div class="outlier-table">
    <h4>📋 Detailed Outliers Information</h4>
    <p><strong>Note:</strong> Outliers are detected using the IQR method (values beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR). MAPE values shown represent individual week performance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display the enhanced table
    display_columns = [
        'Week', 'Location', 'Department', 'Shift', 'Outlier Type',
        'Actual Attendance (%)', 'Greykite Forecast (%)', '4-Week MA Forecast (%)',
        'Greykite MAPE (%)', 'MA MAPE (%)'
    ]
    
    st.dataframe(
        outliers_display[display_columns].sort_values(['Week', 'Location']),
        use_container_width=True,
        height=450
    )
    
    # Additional insights
    if len(outliers_display) > 0:
        worst_greykite = outliers_display.loc[outliers_display['Greykite MAPE (%)'].idxmax()]
        worst_ma = outliers_display.loc[outliers_display['MA MAPE (%)'].idxmax()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **🎯 Worst Greykite Performance:**
            - Week: {worst_greykite['Week']}
            - Location: {worst_greykite['Location']}
            - MAPE: {worst_greykite['Greykite MAPE (%)']}%
            """)
        
        with col2:
            st.markdown(f"""
            **📊 Worst MA Performance:**
            - Week: {worst_ma['Week']}
            - Location: {worst_ma['Location']}
            - MAPE: {worst_ma['MA MAPE (%)']}%
            """)

def create_weekly_mape_trends(df):
    """Create weekly MAPE trends chart."""
    st.markdown("### 📅 WEEKLY MAPE TRENDS")
    
    if len(df) == 0:
        st.warning("⚠️ No data available for the selected filters.")
        return
    
    # Prepare weekly aggregated data
    weekly_data = df.groupby('WEEK_BEGIN').agg({
        'GREYKITE_APE': 'mean',  # Weekly MAPE
        'MA_APE': 'mean',        # Weekly MAPE
        'GREYKITE_WINS': 'mean',
        'PERFORMANCE_IMPROVEMENT': 'mean'
    }).reset_index()
    
    weekly_data.columns = ['WEEK_BEGIN', 'GREYKITE_MAPE', 'MA_MAPE', 'WIN_RATE', 'IMPROVEMENT']
    
    # Create the main chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Weekly MAPE Comparison", "Weekly Win Rate"),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        vertical_spacing=0.1
    )
    
    # MAPE trends
    fig.add_trace(
        go.Scatter(
            x=weekly_data['WEEK_BEGIN'],
            y=weekly_data['GREYKITE_MAPE'],
            mode='lines+markers',
            name='Greykite MAPE',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=weekly_data['WEEK_BEGIN'],
            y=weekly_data['MA_MAPE'],
            mode='lines+markers',
            name='4-Week MA MAPE',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Add performance improvement on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=weekly_data['WEEK_BEGIN'],
            y=weekly_data['IMPROVEMENT'],
            mode='lines',
            name='Performance Improvement (%)',
            line=dict(color='#2ca02c', width=2, dash='dot'),
            yaxis='y2'
        ),
        row=1, col=1, secondary_y=True
    )
    
    # Win rate chart
    fig.add_trace(
        go.Scatter(
            x=weekly_data['WEEK_BEGIN'],
            y=weekly_data['WIN_RATE'] * 100,
            mode='lines+markers',
            name='Win Rate (%)',
            line=dict(color='#d62728', width=3),
            marker=dict(size=8),
            fill='tonexty'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Week", row=2, col=1)
    fig.update_yaxes(title_text="MAPE (%)", row=1, col=1)
    fig.update_yaxes(title_text="Improvement (%)", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Win Rate (%)", row=2, col=1)
    
    fig.update_layout(
        height=700,
        title_text="📊 WEEKLY MAPE PERFORMANCE TRENDS",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_greykite_mape = weekly_data['GREYKITE_MAPE'].mean()
        st.metric("📊 Avg Weekly Greykite MAPE", f"{avg_greykite_mape:.2f}%")
    
    with col2:
        avg_ma_mape = weekly_data['MA_MAPE'].mean()
        st.metric("📈 Avg Weekly MA MAPE", f"{avg_ma_mape:.2f}%")
    
    with col3:
        avg_win_rate = weekly_data['WIN_RATE'].mean() * 100
        st.metric("🏆 Avg Win Rate", f"{avg_win_rate:.1f}%")
    
    with col4:
        avg_improvement = weekly_data['IMPROVEMENT'].mean()
        st.metric("🎯 Avg Improvement", f"{avg_improvement:.2f}%")

def create_large_error_analysis(df):
    """Create large error analysis focusing on MAPE >= 6%."""
    st.markdown("### 🔍 LARGE ERROR ANALYSIS (MAPE ≥ 6%)")
    
    if len(df) == 0:
        st.warning("⚠️ No data available for the selected filters.")
        return
    
    # Filter for large errors (MAPE >= 6%)
    large_errors_greykite = df[df['GREYKITE_APE'] >= 6].copy()
    large_errors_ma = df[df['MA_APE'] >= 6].copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🎯 Greykite Large Errors", len(large_errors_greykite))
        if len(large_errors_greykite) > 0:
            avg_large_mape = large_errors_greykite['GREYKITE_APE'].mean()
            st.metric("📊 Avg MAPE (Large Errors)", f"{avg_large_mape:.2f}%")
    
    with col2:
        st.metric("📈 MA Large Errors", len(large_errors_ma))
        if len(large_errors_ma) > 0:
            avg_large_mape_ma = large_errors_ma['MA_APE'].mean()
            st.metric("📊 Avg MAPE (Large Errors)", f"{avg_large_mape_ma:.2f}%")
    
    # Analysis by segments
    if len(large_errors_greykite) > 0 or len(large_errors_ma) > 0:
        st.markdown("#### 📊 Large Errors by Segment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if len(large_errors_greykite) > 0:
                st.markdown("**🎯 Greykite Large Errors by Location:**")
                location_errors = large_errors_greykite.groupby('WORK_LOCATION').agg({
                    'GREYKITE_APE': ['count', 'mean', 'max']
                }).round(2)
                location_errors.columns = ['Count', 'Avg MAPE (%)', 'Max MAPE (%)']
                st.dataframe(location_errors, use_container_width=True)
        
        with col2:
            if len(large_errors_ma) > 0:
                st.markdown("**📈 MA Large Errors by Location:**")
                location_errors_ma = large_errors_ma.groupby('WORK_LOCATION').agg({
                    'MA_APE': ['count', 'mean', 'max']
                }).round(2)
                location_errors_ma.columns = ['Count', 'Avg MAPE (%)', 'Max MAPE (%)']
                st.dataframe(location_errors_ma, use_container_width=True)
        
        # Large errors distribution
        if len(large_errors_greykite) > 0 and len(large_errors_ma) > 0:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Greykite Large Errors Distribution", "MA Large Errors Distribution"]
            )
            
            fig.add_trace(
                go.Histogram(
                    x=large_errors_greykite['GREYKITE_APE'],
                    name='Greykite MAPE ≥ 6%',
                    nbinsx=20,
                    marker_color='#1f77b4',
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Histogram(
                    x=large_errors_ma['MA_APE'],
                    name='MA MAPE ≥ 6%',
                    nbinsx=20,
                    marker_color='#ff7f0e',
                    opacity=0.7
                ),
                row=1, col=2
            )
            
            fig.update_layout(height=400, title_text="📊 LARGE ERRORS DISTRIBUTION")
            st.plotly_chart(fig, use_container_width=True)

def create_forecast_accuracy_chart(df):
    """Create forecast accuracy analysis chart."""
    st.markdown("### 📈 FORECAST ACCURACY ANALYSIS")
    
    if len(df) == 0:
        st.warning("⚠️ No data available for the selected filters.")
        return
    
    # Create accuracy scatter plots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Greykite: Actual vs Predicted", "4-Week MA: Actual vs Predicted"]
    )
    
    # Greykite scatter plot
    fig.add_trace(
        go.Scatter(
            x=df['ACTUAL_ATTENDANCE_RATE'],
            y=df['GREYKITE_FORECAST'],
            mode='markers',
            name='Greykite',
            marker=dict(
                color=df['GREYKITE_APE'],
                colorscale='RdYlBu_r',
                size=8,
                colorbar=dict(title="MAPE (%)", x=0.45),
                opacity=0.7
            ),
                         text=df.apply(lambda row: f"{row['WEEK_NUMBER']}<br>MAPE: {row['GREYKITE_APE']:.2f}%", axis=1),
            hovertemplate="<b>%{text}</b><br>Actual: %{x:.1f}%<br>Predicted: %{y:.1f}%<extra></extra>"
        ),
        row=1, col=1
    )
    
    # MA scatter plot
    fig.add_trace(
        go.Scatter(
            x=df['ACTUAL_ATTENDANCE_RATE'],
            y=df['MOVING_AVG_4WEEK_FORECAST'],
            mode='markers',
            name='4-Week MA',
            marker=dict(
                color=df['MA_APE'],
                colorscale='RdYlBu_r',
                size=8,
                colorbar=dict(title="MAPE (%)", x=1.02),
                opacity=0.7
            ),
                         text=df.apply(lambda row: f"{row['WEEK_NUMBER']}<br>MAPE: {row['MA_APE']:.2f}%", axis=1),
            hovertemplate="<b>%{text}</b><br>Actual: %{x:.1f}%<br>Predicted: %{y:.1f}%<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Add perfect prediction lines (y=x)
    min_val = df['ACTUAL_ATTENDANCE_RATE'].min()
    max_val = df['ACTUAL_ATTENDANCE_RATE'].max()
    
    for col in [1, 2]:
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash', width=2),
                showlegend=(col == 1)
            ),
            row=1, col=col
        )
    
    # Update layout
    fig.update_xaxes(title_text="Actual Attendance Rate (%)")
    fig.update_yaxes(title_text="Predicted Attendance Rate (%)")
    
    fig.update_layout(
        height=600,
        title_text="📊 FORECAST ACCURACY WITH MAPE COLOR CODING",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    col1, col2 = st.columns(2)
    
    with col1:
        greykite_corr = df['ACTUAL_ATTENDANCE_RATE'].corr(df['GREYKITE_FORECAST'])
        st.metric("🎯 Greykite Correlation", f"{greykite_corr:.3f}")
        
        greykite_rmse = np.sqrt(df['GREYKITE_SE'].mean())
        st.metric("📊 Greykite RMSE", f"{greykite_rmse:.2f}%")
    
    with col2:
        ma_corr = df['ACTUAL_ATTENDANCE_RATE'].corr(df['MOVING_AVG_4WEEK_FORECAST'])
        st.metric("📈 MA Correlation", f"{ma_corr:.3f}")
        
        ma_rmse = np.sqrt(df['MA_SE'].mean())
        st.metric("📊 MA RMSE", f"{ma_rmse:.2f}%")

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
    
    create_weekly_mape_trends(filtered_df)
    st.markdown("---")
    
    create_large_error_analysis(filtered_df)
    st.markdown("---")
    
    create_forecast_accuracy_chart(filtered_df)
    
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
