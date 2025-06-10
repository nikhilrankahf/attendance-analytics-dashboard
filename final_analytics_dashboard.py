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
    
    return filters

def apply_filters(df, filters):
    """Apply selected filters to the dataframe."""
    filtered_df = df.copy()
    
    # Apply filters
    if filters['years']:
        filtered_df = filtered_df[filtered_df['YEAR'].isin(filters['years'])]
    if filters['locations']:
        filtered_df = filtered_df[filtered_df['WORK_LOCATION'].isin(filters['locations'])]
    if filters['departments']:
        filtered_df = filtered_df[filtered_df['DEPARTMENT_GROUP'].isin(filters['departments'])]
    if filters['shifts']:
        filtered_df = filtered_df[filtered_df['SHIFT_TIME'].isin(filters['shifts'])]
    
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
    """Create clean executive summary with key insights only - NO FILTERED DATA INSIGHTS."""
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
    
    # SINGLE COLUMN LAYOUT - NO "FILTERED DATA INSIGHTS"
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
    
    # Create enhanced 3-column metrics display - NO RMSE OR AVG PERFORMANCE GAIN
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
        'WEEK_BEGIN', 'WORK_LOCATION', 'DEPARTMENT_GROUP', 'SHIFT_TIME',
        'ACTUAL_ATTENDANCE_RATE', 'GREYKITE_FORECAST', 'MOVING_AVG_4WEEK_FORECAST',
        'GREYKITE_APE', 'MA_APE', 'GREYKITE_APE_OUTLIER', 'MA_APE_OUTLIER'
    ]].copy()
    
    # Format the data for better display
    outliers_display['WEEK_BEGIN'] = outliers_display['WEEK_BEGIN'].dt.strftime('%Y-%m-%d')
    outliers_display['ACTUAL_ATTENDANCE_RATE'] = outliers_display['ACTUAL_ATTENDANCE_RATE'].round(2)
    outliers_display['GREYKITE_FORECAST'] = outliers_display['GREYKITE_FORECAST'].round(2)
    outliers_display['MOVING_AVG_4WEEK_FORECAST'] = outliers_display['MOVING_AVG_4WEEK_FORECAST'].round(2)
    outliers_display['GREYKITE_APE'] = outliers_display['GREYKITE_APE'].round(2)
    outliers_display['MA_APE'] = outliers_display['MA_APE'].round(2)
    
    # Rename columns for better readability
    outliers_display.columns = [
        'Week', 'Location', 'Department', 'Shift',
        'Actual Attendance (%)', 'Greykite Forecast (%)', '4-Week MA Forecast (%)',
        'Greykite APE (%)', 'MA APE (%)', 'Greykite Outlier', 'MA Outlier'
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
    <p><strong>Note:</strong> Outliers are detected using the IQR method (values beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display the enhanced table
    display_columns = [
        'Week', 'Location', 'Department', 'Shift', 'Outlier Type',
        'Actual Attendance (%)', 'Greykite Forecast (%)', '4-Week MA Forecast (%)',
        'Greykite APE (%)', 'MA APE (%)'
    ]
    
    st.dataframe(
        outliers_display[display_columns].sort_values(['Week', 'Location']),
        use_container_width=True,
        height=450
    )
    
    # Additional insights
    if len(outliers_display) > 0:
        worst_greykite = outliers_display.loc[outliers_display['Greykite APE (%)'].idxmax()]
        worst_ma = outliers_display.loc[outliers_display['MA APE (%)'].idxmax()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **🎯 Worst Greykite Performance:**
            - Week: {worst_greykite['Week']}
            - Location: {worst_greykite['Location']}
            - APE: {worst_greykite['Greykite APE (%)']}%
            """)
        
        with col2:
            st.markdown(f"""
            **📊 Worst MA Performance:**
            - Week: {worst_ma['Week']}
            - Location: {worst_ma['Location']}
            - APE: {worst_ma['MA APE (%)']}%
            """)

def create_weekly_mape_trends(df):
    """Create weekly MAPE trends visualization comparing Greykite vs 4-week moving average."""
    st.markdown("### 📈 WEEKLY MAPE TRENDS ANALYSIS")
    
    if len(df) == 0:
        st.warning("⚠️ No data available for the selected filters.")
        return
    
    # Prepare weekly aggregated data
    weekly_trends = df.groupby('WEEK_BEGIN').agg({
        'GREYKITE_APE': 'mean',
        'MA_APE': 'mean',
        'GREYKITE_WINS': 'sum',
        'ACTUAL_ATTENDANCE_RATE': 'mean',
        'GREYKITE_FORECAST': 'mean',
        'MOVING_AVG_4WEEK_FORECAST': 'mean'
    }).reset_index()
    
    weekly_trends['WEEK_COUNT'] = df.groupby('WEEK_BEGIN').size().values
    weekly_trends['WIN_RATE'] = (weekly_trends['GREYKITE_WINS'] / weekly_trends['WEEK_COUNT']) * 100
    
    # Create the trend chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "📊 Weekly MAPE Comparison: Greykite vs 4-Week Moving Average",
            "🏆 Weekly Win Rate & Data Points"
        ],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
        vertical_spacing=0.12
    )
    
    # MAPE trends
    fig.add_trace(
        go.Scatter(
            x=weekly_trends['WEEK_BEGIN'],
            y=weekly_trends['GREYKITE_APE'],
            name='Greykite MAPE',
            line=dict(color='#1f77b4', width=3),
            mode='lines+markers',
            marker=dict(size=6),
            hovertemplate='<b>Greykite MAPE</b><br>Week: %{x}<br>MAPE: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=weekly_trends['WEEK_BEGIN'],
            y=weekly_trends['MA_APE'],
            name='4-Week MA MAPE',
            line=dict(color='#ff7f0e', width=3),
            mode='lines+markers',
            marker=dict(size=6),
            hovertemplate='<b>4-Week MA MAPE</b><br>Week: %{x}<br>MAPE: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Win rate and data volume
    fig.add_trace(
        go.Scatter(
            x=weekly_trends['WEEK_BEGIN'],
            y=weekly_trends['WIN_RATE'],
            name='Win Rate %',
            line=dict(color='#2ca02c', width=2),
            mode='lines+markers',
            marker=dict(size=5),
            hovertemplate='<b>Win Rate</b><br>Week: %{x}<br>Rate: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Data volume as bar chart
    fig.add_trace(
        go.Bar(
            x=weekly_trends['WEEK_BEGIN'],
            y=weekly_trends['WEEK_COUNT'],
            name='Data Points',
            marker_color='lightblue',
            opacity=0.6,
            yaxis='y4',
            hovertemplate='<b>Data Points</b><br>Week: %{x}<br>Count: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        title_text="📈 COMPREHENSIVE WEEKLY MAPE TRENDS ANALYSIS",
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="MAPE (%)", row=1, col=1)
    fig.update_yaxes(title_text="Win Rate (%)", row=2, col=1)
    fig.update_yaxes(title_text="Data Points", side="right", row=2, col=1)
    fig.update_xaxes(title_text="Week", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary insights
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_greykite_mape = weekly_trends['GREYKITE_APE'].mean()
        st.metric("📊 Avg Weekly Greykite MAPE", f"{avg_greykite_mape:.2f}%")
    with col2:
        avg_ma_mape = weekly_trends['MA_APE'].mean()
        st.metric("📈 Avg Weekly MA MAPE", f"{avg_ma_mape:.2f}%")
    with col3:
        avg_win_rate = weekly_trends['WIN_RATE'].mean()
        st.metric("🏆 Avg Weekly Win Rate", f"{avg_win_rate:.1f}%")

def create_large_error_analysis(df):
    """Create large error analysis for Greykite (+6 percentage points difference)."""
    st.markdown("### 🚨 LARGE ERROR ANALYSIS (Greykite)")
    
    if len(df) == 0:
        st.warning("⚠️ No data available for the selected filters.")
        return
    
    # Define large errors as +6 percentage points difference
    large_errors_df = df[df['GREYKITE_APE'] >= 6.0].copy()
    
    if len(large_errors_df) == 0:
        st.info("✅ No large errors (≥6% APE) detected for Greykite in the current dataset.")
        return
    
    # Calculate error categories
    df_analysis = df.copy()
    df_analysis['ERROR_CATEGORY'] = pd.cut(
        df_analysis['GREYKITE_APE'], 
        bins=[0, 2, 4, 6, 10, float('inf')], 
        labels=['Excellent (0-2%)', 'Good (2-4%)', 'Moderate (4-6%)', 'Poor (6-10%)', 'Very Poor (>10%)'],
        include_lowest=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Error distribution pie chart
        error_dist = df_analysis['ERROR_CATEGORY'].value_counts()
        fig_pie = px.pie(
            values=error_dist.values,
            names=error_dist.index,
            title="🎯 Greykite Error Distribution",
            color_discrete_sequence=['#28a745', '#90ee90', '#ffc107', '#fd7e14', '#dc3545']
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Large errors by location
        if len(large_errors_df) > 0:
            location_errors = large_errors_df.groupby('WORK_LOCATION').agg({
                'GREYKITE_APE': ['count', 'mean']
            }).round(2)
            location_errors.columns = ['Error_Count', 'Avg_APE']
            location_errors = location_errors.sort_values('Error_Count', ascending=True)
            
            fig_bar = px.bar(
                x=location_errors['Error_Count'],
                y=location_errors.index,
                orientation='h',
                title="🏢 Large Errors by Location",
                labels={'x': 'Number of Large Errors', 'y': 'Location'},
                color=location_errors['Avg_APE'],
                color_continuous_scale='Reds'
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Large errors timeline
    if len(large_errors_df) > 0:
        timeline_errors = large_errors_df.groupby('WEEK_BEGIN').agg({
            'GREYKITE_APE': ['count', 'mean', 'max'],
            'ACTUAL_ATTENDANCE_RATE': 'mean',
            'GREYKITE_FORECAST': 'mean'
        }).reset_index()
        timeline_errors.columns = ['WEEK_BEGIN', 'Error_Count', 'Avg_APE', 'Max_APE', 'Actual_Rate', 'Forecast']
        
        fig_timeline = make_subplots(
            rows=2, cols=1,
            subplot_titles=["📅 Large Errors Timeline", "📊 Error Severity & Frequency"],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        # Timeline of large errors
        fig_timeline.add_trace(
            go.Scatter(
                x=timeline_errors['WEEK_BEGIN'],
                y=timeline_errors['Error_Count'],
                name='Error Count',
                line=dict(color='red', width=3),
                mode='lines+markers'
            ),
            row=1, col=1
        )
        
        fig_timeline.add_trace(
            go.Scatter(
                x=timeline_errors['WEEK_BEGIN'],
                y=timeline_errors['Avg_APE'],
                name='Avg APE',
                line=dict(color='orange', width=2),
                mode='lines+markers',
                yaxis='y2'
            ),
            row=1, col=1
        )
        
        # Error severity distribution
        fig_timeline.add_trace(
            go.Bar(
                x=timeline_errors['WEEK_BEGIN'],
                y=timeline_errors['Max_APE'],
                name='Max APE',
                marker_color='darkred',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig_timeline.update_yaxes(title_text="Error Count", row=1, col=1)
        fig_timeline.update_yaxes(title_text="Average APE (%)", side="right", row=1, col=1)
        fig_timeline.update_yaxes(title_text="Maximum APE (%)", row=2, col=1)
        fig_timeline.update_xaxes(title_text="Week", row=2, col=1)
        
        fig_timeline.update_layout(height=600, title_text="🚨 LARGE ERROR TRENDS ANALYSIS")
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        large_error_rate = (len(large_errors_df) / len(df)) * 100
        st.metric("🚨 Large Error Rate", f"{large_error_rate:.1f}%")
    with col2:
        avg_large_error = large_errors_df['GREYKITE_APE'].mean()
        st.metric("📊 Avg Large Error APE", f"{avg_large_error:.1f}%")
    with col3:
        max_error = large_errors_df['GREYKITE_APE'].max()
        st.metric("⚠️ Maximum Error", f"{max_error:.1f}%")
    with col4:
        most_affected_location = large_errors_df['WORK_LOCATION'].mode().iloc[0] if len(large_errors_df) > 0 else "N/A"
        st.metric("🏢 Most Affected Location", most_affected_location)

def create_forecast_accuracy_chart(df):
    """Create forecast accuracy visualization - actual vs Greykite forecast."""
    st.markdown("### 🎯 FORECAST ACCURACY ANALYSIS")
    
    if len(df) == 0:
        st.warning("⚠️ No data available for the selected filters.")
        return
    
    # Clean data by dropping rows with NaN values in key columns
    clean_df = df.dropna(subset=['GREYKITE_APE', 'ACTUAL_ATTENDANCE_RATE', 'GREYKITE_FORECAST']).copy()
    
    if len(clean_df) == 0:
        st.warning("⚠️ No valid data available after removing NaN values.")
        return
    
    # Sample data for visualization if too many points
    display_df = clean_df.sample(n=min(1000, len(clean_df))).copy() if len(clean_df) > 1000 else clean_df.copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter plot: Actual vs Predicted
        fig_scatter = px.scatter(
            display_df,
            x='ACTUAL_ATTENDANCE_RATE',
            y='GREYKITE_FORECAST',
            color='GREYKITE_APE',
            size='GREYKITE_APE',
            hover_data=['WORK_LOCATION', 'DEPARTMENT_GROUP', 'WEEK_BEGIN'],
            title="📊 Actual vs Greykite Forecast",
            labels={
                'ACTUAL_ATTENDANCE_RATE': 'Actual Attendance Rate (%)',
                'GREYKITE_FORECAST': 'Greykite Forecast (%)',
                'GREYKITE_APE': 'APE (%)'
            },
            color_continuous_scale='RdYlBu_r'
        )
        
        # Add perfect prediction line
        min_val = min(display_df['ACTUAL_ATTENDANCE_RATE'].min(), display_df['GREYKITE_FORECAST'].min())
        max_val = max(display_df['ACTUAL_ATTENDANCE_RATE'].max(), display_df['GREYKITE_FORECAST'].max())
        fig_scatter.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', width=2, dash='dash')
            )
        )
        
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Residuals plot
        display_df['RESIDUALS'] = display_df['GREYKITE_FORECAST'] - display_df['ACTUAL_ATTENDANCE_RATE']
        
        fig_residuals = px.scatter(
            display_df,
            x='ACTUAL_ATTENDANCE_RATE',
            y='RESIDUALS',
            color='GREYKITE_APE',
            title="📈 Residuals Analysis",
            labels={
                'ACTUAL_ATTENDANCE_RATE': 'Actual Attendance Rate (%)',
                'RESIDUALS': 'Residuals (Forecast - Actual)',
                'GREYKITE_APE': 'APE (%)'
            },
            color_continuous_scale='RdYlBu_r'
        )
        
        # Add zero line
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
        fig_residuals.update_layout(height=500)
        st.plotly_chart(fig_residuals, use_container_width=True)
    
    # Accuracy metrics over time - use clean_df instead of df
    time_accuracy = clean_df.groupby('WEEK_BEGIN').agg({
        'GREYKITE_APE': 'mean',
        'GREYKITE_ABS_ERROR': 'mean',
        'ACTUAL_ATTENDANCE_RATE': 'mean',
        'GREYKITE_FORECAST': 'mean',
        'GREYKITE_WINS': 'sum'
    }).reset_index()
    
    time_accuracy['WEEK_COUNT'] = clean_df.groupby('WEEK_BEGIN').size().values
    time_accuracy['WIN_RATE'] = (time_accuracy['GREYKITE_WINS'] / time_accuracy['WEEK_COUNT']) * 100
    time_accuracy['BIAS'] = time_accuracy['GREYKITE_FORECAST'] - time_accuracy['ACTUAL_ATTENDANCE_RATE']
    
    fig_time = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "📊 Weekly Average APE", "🎯 Weekly Bias (Forecast - Actual)",
            "📈 Actual vs Forecast Trends", "🏆 Weekly Win Rate"
        ]
    )
    
    # Weekly APE
    fig_time.add_trace(
        go.Scatter(x=time_accuracy['WEEK_BEGIN'], y=time_accuracy['GREYKITE_APE'],
                  name='APE', line=dict(color='blue')), row=1, col=1)
    
    # Weekly Bias
    fig_time.add_trace(
        go.Scatter(x=time_accuracy['WEEK_BEGIN'], y=time_accuracy['BIAS'],
                  name='Bias', line=dict(color='red')), row=1, col=2)
    fig_time.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
    
    # Actual vs Forecast trends
    fig_time.add_trace(
        go.Scatter(x=time_accuracy['WEEK_BEGIN'], y=time_accuracy['ACTUAL_ATTENDANCE_RATE'],
                  name='Actual', line=dict(color='green')), row=2, col=1)
    fig_time.add_trace(
        go.Scatter(x=time_accuracy['WEEK_BEGIN'], y=time_accuracy['GREYKITE_FORECAST'],
                  name='Forecast', line=dict(color='orange')), row=2, col=1)
    
    # Win rate
    fig_time.add_trace(
        go.Scatter(x=time_accuracy['WEEK_BEGIN'], y=time_accuracy['WIN_RATE'],
                  name='Win Rate', line=dict(color='purple')), row=2, col=2)
    
    fig_time.update_layout(height=600, title_text="🎯 COMPREHENSIVE FORECAST ACCURACY ANALYSIS")
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Summary accuracy metrics - use clean_df
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        mae = clean_df['GREYKITE_ABS_ERROR'].mean()
        st.metric("📊 Mean Absolute Error", f"{mae:.2f}%")
    with col2:
        rmse = np.sqrt(clean_df['GREYKITE_SE'].mean())
        st.metric("📈 Root Mean Square Error", f"{rmse:.2f}%")
    with col3:
        correlation = clean_df['ACTUAL_ATTENDANCE_RATE'].corr(clean_df['GREYKITE_FORECAST'])
        st.metric("🔗 Correlation", f"{correlation:.3f}")
    with col4:
        bias = clean_df['GREYKITE_ERROR'].mean()
        st.metric("⚖️ Bias", f"{bias:.2f}%")
    
    # Add data quality info
    if len(clean_df) < len(df):
        st.info(f"ℹ️ Removed {len(df) - len(clean_df)} rows with missing values. Showing analysis for {len(clean_df)} valid records.")

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