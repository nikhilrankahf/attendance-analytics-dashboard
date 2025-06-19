import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt
from datetime import datetime, timedelta
import os
import glob
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
    """Load and process the attendance forecast data with enhanced Greykite models."""
    try:
        # Dynamic file detection for latest forecast output
        files = glob.glob('enhanced_attendance_forecast_all_combinations_2024_2025_*.csv')
        if not files:
            st.error("⚠️ No enhanced forecast files found! Looking for pattern: enhanced_attendance_forecast_all_combinations_2024_2025_*.csv")
            return None
        
        # Get the most recent file
        latest_file = max(files, key=os.path.getctime)
        st.info(f"📊 Loading data from: {latest_file}")
        df = pd.read_csv(latest_file)
        
        # Column mapping from new format to dashboard expectations
        column_mapping = {
            'week_number': 'WEEK_NUMBER',
            'work_location': 'WORK_LOCATION', 
            'shift_time': 'SHIFT_TIME',
            'department_group': 'DEPARTMENT_GROUP',
            'actual_attendance': 'ACTUAL_ATTENDANCE_RATE',
            'greykite_forecast': 'GREYKITE_FORECAST',
            'four_week_rolling_avg': 'MOVING_AVG_4WEEK_FORECAST',
            'six_week_rolling_avg': 'SIX_WEEK_ROLLING_AVG',
            'exponential_smoothing_alpha_2': 'EXP_SMOOTH_02',
            'exponential_smoothing_alpha_4': 'EXP_SMOOTH_04',
            'exponential_smoothing_alpha_6': 'EXP_SMOOTH_06', 
            'exponential_smoothing_alpha_8': 'EXP_SMOOTH_08',
            'exponential_smoothing_alpha_1': 'EXP_SMOOTH_10',
            'greykite_forecast_lower_95': 'GREYKITE_FORECAST_LOWER',
            'greykite_forecast_upper_95': 'GREYKITE_FORECAST_UPPER'
        }
        
        # Rename columns to match dashboard expectations
        df = df.rename(columns=column_mapping)
        
        # Extract year from week_number for filtering (e.g., 2024-W01 -> 2024)
        def extract_year_from_week(week_str):
            try:
                if pd.isna(week_str) or week_str == '':
                    return None
                year = str(week_str).split('-W')[0]
                return int(year)
            except:
                return None
        
        def extract_week_num_from_week(week_str):
            try:
                if pd.isna(week_str) or week_str == '':
                    return None
                week = str(week_str).split('-W')[1]
                return int(week)
            except:
                return None
        
        df['YEAR'] = df['WEEK_NUMBER'].apply(extract_year_from_week)
        df['WEEK_NUM'] = df['WEEK_NUMBER'].apply(extract_week_num_from_week)
        
        # Create quarter from week number (approximate)
        df['QUARTER'] = df['WEEK_NUM'].apply(lambda x: 1 if x <= 13 else 2 if x <= 26 else 3 if x <= 39 else 4 if pd.notna(x) else None)
        df['MONTH'] = df['WEEK_NUM'].apply(lambda x: ((x-1) // 4) + 1 if pd.notna(x) and x <= 52 else None)
        
        # Handle missing values in WEEK_NUMBER
        df['WEEK_NUMBER'] = df['WEEK_NUMBER'].fillna('').astype(str)
        df['WEEK_NUMBER_MISSING'] = (df['WEEK_NUMBER'] == '') | (df['WEEK_NUMBER'].isna())
        
        # Create month names and quarter names from numeric values
        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                      7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
        df['MONTH_NAME'] = df['MONTH'].map(month_names)
        df['QUARTER_NAME'] = df.apply(lambda row: f"Q{row['QUARTER']} {row['YEAR']}" if pd.notna(row['QUARTER']) and pd.notna(row['YEAR']) else None, axis=1)
        
        # Enhanced model definitions with multiple exponential smoothing variants
        models = {
            'GREYKITE': 'GREYKITE_FORECAST',
            'MA_4WEEK': 'MOVING_AVG_4WEEK_FORECAST', 
            'MA_6WEEK': 'SIX_WEEK_ROLLING_AVG',
            'EXP_SMOOTH_02': 'EXP_SMOOTH_02',
            'EXP_SMOOTH_04': 'EXP_SMOOTH_04',
            'EXP_SMOOTH_06': 'EXP_SMOOTH_06',
            'EXP_SMOOTH_08': 'EXP_SMOOTH_08',
            'EXP_SMOOTH_10': 'EXP_SMOOTH_10'
        }
        
        # For backward compatibility, create a single EXP_SMOOTH column using alpha=0.4 (best performing)
        if 'EXP_SMOOTH_04' in df.columns:
            df['EXPONENTIAL_SMOOTHING'] = df['EXP_SMOOTH_04']
            models['EXP_SMOOTH'] = 'EXPONENTIAL_SMOOTHING'
        
        # Store original dataframe for conditional aggregation
        original_df = df.copy()
        
        # Note: Aggregation and metric calculation will be handled in apply_filters() based on shift selection
        
        return df
        
    except FileNotFoundError:
        st.error("⚠️ File 'attendance_forecast_results_20250609_191409.csv' not found!")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading data: {str(e)}")
        return None

def calculate_performance_metrics(df):
    """Calculate performance metrics for all models after aggregation."""
    import numpy as np
    
    # Handle case where df might be a Series (convert to DataFrame)
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    
    # Ensure we have a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame or Series, got {type(df)}")
    
    # Check if df has the columns attribute (should now be guaranteed)
    available_cols = df.columns.tolist() if hasattr(df, 'columns') else []
    
    # Define model mappings
    models = {
        'GREYKITE': 'GREYKITE_FORECAST',
        'MA_4WEEK': 'MOVING_AVG_4WEEK_FORECAST',
        'MA_6WEEK': 'SIX_WEEK_ROLLING_AVG',
        'EXP_SMOOTH_02': 'EXP_SMOOTH_02',
        'EXP_SMOOTH_04': 'EXP_SMOOTH_04',
        'EXP_SMOOTH_06': 'EXP_SMOOTH_06',
        'EXP_SMOOTH_08': 'EXP_SMOOTH_08',
        'EXP_SMOOTH_10': 'EXP_SMOOTH_10'
    }
    
    # For backward compatibility, create a single EXP_SMOOTH column using alpha=0.4 (best performing)
    if 'EXP_SMOOTH_04' in available_cols:
        df['EXPONENTIAL_SMOOTHING'] = df['EXP_SMOOTH_04']
        models['EXP_SMOOTH'] = 'EXPONENTIAL_SMOOTHING'
    
    # Compute errors and metrics for ALL models
    for model_name, forecast_col in models.items():
        if forecast_col in available_cols:
            # Only calculate metrics if they haven't been calculated during aggregation
            if f'{model_name}_APE' not in available_cols:
                # Basic errors
                df[f'{model_name}_ERROR'] = df[forecast_col] - df['ACTUAL_ATTENDANCE_RATE']
                df[f'{model_name}_ABS_ERROR'] = np.abs(df[f'{model_name}_ERROR'])
                
                # Weekly MAPE calculation: |Forecast - Actual| / |Actual| * 100 for each individual week
                df[f'{model_name}_WEEKLY_MAPE'] = np.abs(df[f'{model_name}_ERROR']) / np.abs(df['ACTUAL_ATTENDANCE_RATE']) * 100
                df[f'{model_name}_WEEKLY_MAPE'] = df[f'{model_name}_WEEKLY_MAPE'].replace([np.inf, -np.inf], np.nan)
                
                # Keep APE column for backward compatibility (same as WEEKLY_MAPE)
                df[f'{model_name}_APE'] = df[f'{model_name}_WEEKLY_MAPE']
                
                # Squared errors for MSE/RMSE
                df[f'{model_name}_SE'] = df[f'{model_name}_ERROR'] ** 2
                
                # Update available_cols list for new columns
                available_cols.extend([f'{model_name}_ERROR', f'{model_name}_ABS_ERROR', 
                                     f'{model_name}_WEEKLY_MAPE', f'{model_name}_APE', f'{model_name}_SE'])
            
            # Always calculate outlier detection (using IQR method on weekly MAPE)
            if f'{model_name}_APE' in available_cols:
                Q1 = df[f'{model_name}_APE'].quantile(0.25)
                Q3 = df[f'{model_name}_APE'].quantile(0.75)
                IQR = Q3 - Q1
                df[f'{model_name}_APE_OUTLIER'] = ((df[f'{model_name}_APE'] < (Q1 - 1.5 * IQR)) | 
                                                   (df[f'{model_name}_APE'] > (Q3 + 1.5 * IQR))).astype(int)
                available_cols.append(f'{model_name}_APE_OUTLIER')
    
    # Performance comparison - find best model for each row
    available_models = [model for model in models.keys() if f'{model}_ABS_ERROR' in available_cols]
    if available_models:
        error_cols = [f'{model}_ABS_ERROR' for model in available_models]
        df['BEST_MODEL'] = df[error_cols].idxmin(axis=1).str.replace('_ABS_ERROR', '')
        
        # Create win indicators for each model
        for model in available_models:
            df[f'{model}_WINS'] = (df['BEST_MODEL'] == model).astype(int)
    
    # Legacy columns for backward compatibility (keep existing Greykite vs MA comparison)
    if 'GREYKITE_ABS_ERROR' in available_cols and 'MA_4WEEK_ABS_ERROR' in available_cols:
        df['GREYKITE_WINS'] = (df['GREYKITE_ABS_ERROR'] < df['MA_4WEEK_ABS_ERROR']).astype(int)
    
    return df

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
    """Create enhanced sidebar filters for all dashboard controls."""
    st.sidebar.markdown("## 🎛️ DASHBOARD CONTROLS")
    
    # Year filter
    years = sorted(df['YEAR'].unique())
    selected_years = st.sidebar.multiselect(
        "📅 Select Years",
        years,
        default=years,
        help="Filter data by specific years"
    )
    
    # Location filter
    locations = sorted(df['WORK_LOCATION'].unique())
    selected_locations = st.sidebar.multiselect(
        "🏢 Select Locations",
        locations,
        default=locations,
        help="Filter data by work locations"
    )
    
    # Department filter
    departments = sorted(df['DEPARTMENT_GROUP'].unique())
    selected_departments = st.sidebar.multiselect(
        "🏭 Select Departments",
        departments,
        default=departments,
        help="Filter data by department groups"
    )
    
    # Shift filter
    shifts = sorted(df['SHIFT_TIME'].unique())
    selected_shifts = st.sidebar.multiselect(
        "⏰ Select Shifts",
        shifts,
        default=shifts,
        help="Filter data by shift times"
    )
    
    # Model comparison filter
    st.sidebar.markdown("### 🎯 Model Comparison")
    model_options = [
        'All Models Overview',
        'Greykite vs 4-Week MA', 
        'Greykite vs 6-Week MA',
        'Greykite vs Exp. Smoothing',
        'Moving Average Comparison',
        'Exponential Smoothing Variants',
        'Best vs Worst Performance',
        'Individual Model Focus'
    ]
    selected_comparison = st.sidebar.selectbox(
        "Select Comparison Type",
        model_options,
        help="Choose how to compare the forecasting models"
    )
    
    # Individual model focus (if selected)
    focus_model = None
    if selected_comparison == 'Individual Model Focus':
        model_focus_options = ['Greykite', '4-Week MA', '6-Week MA']
        # Add exponential smoothing variants if available
        if 'EXP_SMOOTH_02' in df.columns:
            model_focus_options.extend(['Exp. Smooth α=0.2', 'Exp. Smooth α=0.4', 'Exp. Smooth α=0.6', 'Exp. Smooth α=0.8', 'Exp. Smooth α=1.0'])
        focus_model = st.sidebar.selectbox(
            "Focus on Model",
            model_focus_options,
            help="Select a specific model to analyze in detail"
        )
    
    # Performance filter
    performance_options = ["All Data", "High Performance (MAPE < 3%)", "Medium Performance (3% ≤ MAPE < 6%)", "Low Performance (MAPE ≥ 6%)"]
    selected_performance = st.sidebar.selectbox(
        "📊 Performance Filter",
        performance_options,
        help="Filter data by forecast accuracy levels"
    )
    
    # Week range filter
    st.sidebar.markdown("### 📅 Week Range")
    available_weeks = sorted([w for w in df['WEEK_NUMBER'].unique() if w and w != ''])
    
    if available_weeks:
        week_range = st.sidebar.select_slider(
            "Select Week Range",
            options=available_weeks,
            value=(available_weeks[0], available_weeks[-1]),
            help="Filter data by specific week range"
        )
    else:
        week_range = None
    
    return {
        'years': selected_years,
        'locations': selected_locations,
        'departments': selected_departments,
        'shifts': selected_shifts,
        'comparison': selected_comparison,
        'focus_model': focus_model,
        'performance': selected_performance,
        'week_range': week_range
    }

def apply_filters(df, filters):
    """Apply all selected filters to the dataframe with conditional AM/PM aggregation."""
    filtered_df = df.copy()
    
    # Apply basic filters first
    if filters['years']:
        filtered_df = filtered_df[filtered_df['YEAR'].isin(filters['years'])]
    
    if filters['locations']:
        filtered_df = filtered_df[filtered_df['WORK_LOCATION'].isin(filters['locations'])]
    
    if filters['departments']:
        filtered_df = filtered_df[filtered_df['DEPARTMENT_GROUP'].isin(filters['departments'])]
    
    # Apply week range filter
    if filters['week_range'] and len(filters['week_range']) == 2:
        start_week, end_week = filters['week_range']
        # Filter by week range (string comparison works for YYYY-WXX format)
        filtered_df = filtered_df[
            (filtered_df['WEEK_NUMBER'] >= start_week) &
            (filtered_df['WEEK_NUMBER'] <= end_week)
        ]
    
    # CONDITIONAL AGGREGATION LOGIC:
    # If both AM and PM shifts are selected (or all shifts), aggregate them
    # If only one shift is selected, show individual shift data
    
    all_shifts = sorted(df['SHIFT_TIME'].unique())
    selected_shifts = filters.get('shifts', all_shifts)
    
    if len(selected_shifts) > 1 and set(selected_shifts) == set(all_shifts):
        # Both AM and PM selected - aggregate by week/location/department
        st.info("🔄 Aggregating AM/PM shifts by week/location/department for combined view...")
        
        def aggregate_shifts(group):
            """Aggregate AM/PM shifts for each week/location/department combination with correct MAPE calculation"""
            import numpy as np
            import pandas as pd
            
            # Ensure group is a DataFrame (handle case where groupby returns Series)
            if isinstance(group, pd.Series):
                group = group.to_frame().T
            
            # Ensure we have a DataFrame
            if not isinstance(group, pd.DataFrame):
                raise TypeError(f"Expected DataFrame or Series, got {type(group)}")
            
            result = {}
            
            # Ensure result is always a dictionary
            if not isinstance(result, dict):
                raise TypeError(f"Result should be a dictionary, got {type(result)}")
            
            # Take the first values for grouping columns (they're the same within group)
            result['WEEK_NUMBER'] = group['WEEK_NUMBER'].iloc[0]
            result['WORK_LOCATION'] = group['WORK_LOCATION'].iloc[0] 
            result['DEPARTMENT_GROUP'] = group['DEPARTMENT_GROUP'].iloc[0]
            result['YEAR'] = group['YEAR'].iloc[0]
            result['WEEK_NUM'] = group['WEEK_NUM'].iloc[0]
            result['QUARTER'] = group['QUARTER'].iloc[0]
            result['MONTH'] = group['MONTH'].iloc[0]
            result['MONTH_NAME'] = group['MONTH_NAME'].iloc[0]
            result['QUARTER_NAME'] = group['QUARTER_NAME'].iloc[0]
            result['WEEK_NUMBER_MISSING'] = group['WEEK_NUMBER_MISSING'].iloc[0]
            
            # For attendance and forecast columns, take the mean of AM/PM
            numeric_cols = ['ACTUAL_ATTENDANCE_RATE', 'GREYKITE_FORECAST', 'MOVING_AVG_4WEEK_FORECAST', 
                           'SIX_WEEK_ROLLING_AVG', 'EXP_SMOOTH_02', 'EXP_SMOOTH_04', 'EXP_SMOOTH_06', 
                           'EXP_SMOOTH_08', 'EXP_SMOOTH_10', 'EXPONENTIAL_SMOOTHING',
                           'GREYKITE_FORECAST_LOWER', 'GREYKITE_FORECAST_UPPER']
            
            # Get available columns (works for both DataFrame and Series)
            if hasattr(group, 'columns'):
                available_cols = group.columns.tolist()
            elif hasattr(group, 'index'):
                available_cols = group.index.tolist()
            else:
                available_cols = []
            
            for col in numeric_cols:
                if col in available_cols:
                    result[col] = group[col].mean()
            
            # Calculate individual APE values for each shift, then average them (correct MAPE formula)
            # This ensures MAPE = mean(|forecast - actual| / |actual|) × 100
            models = {
                'GREYKITE': 'GREYKITE_FORECAST',
                'MA_4WEEK': 'MOVING_AVG_4WEEK_FORECAST',
                'MA_6WEEK': 'SIX_WEEK_ROLLING_AVG',
                'EXP_SMOOTH_02': 'EXP_SMOOTH_02',
                'EXP_SMOOTH_04': 'EXP_SMOOTH_04',
                'EXP_SMOOTH_06': 'EXP_SMOOTH_06',
                'EXP_SMOOTH_08': 'EXP_SMOOTH_08',
                'EXP_SMOOTH_10': 'EXP_SMOOTH_10'
            }
            
            # For backward compatibility
            if 'EXP_SMOOTH_04' in available_cols:
                models['EXP_SMOOTH'] = 'EXP_SMOOTH_04'  # Use alpha=0.4 as representative
            
            for model_name, forecast_col in models.items():
                if forecast_col in available_cols and 'ACTUAL_ATTENDANCE_RATE' in available_cols:
                    # Calculate APE for each individual shift
                    individual_apes = []
                    for idx in group.index:
                        actual = group.loc[idx, 'ACTUAL_ATTENDANCE_RATE']
                        forecast = group.loc[idx, forecast_col]
                        if pd.notna(actual) and pd.notna(forecast) and actual != 0:
                            ape = abs(forecast - actual) / abs(actual) * 100
                            individual_apes.append(ape)
                    
                    # Store the mean of individual APEs (correct MAPE calculation)
                    if individual_apes:
                        # Ensure result is still a dictionary
                        if not isinstance(result, dict):
                            raise TypeError(f"Result should be a dictionary at APE calculation, got {type(result)}")
                        
                        result[f'{model_name}_APE'] = np.mean(individual_apes)
                        result[f'{model_name}_WEEKLY_MAPE'] = np.mean(individual_apes)
                        
                        # Also calculate other metrics based on individual shift data
                        individual_errors = []
                        individual_abs_errors = []
                        individual_se = []
                        
                        for idx in group.index:
                            actual = group.loc[idx, 'ACTUAL_ATTENDANCE_RATE']
                            forecast = group.loc[idx, forecast_col]
                            if pd.notna(actual) and pd.notna(forecast):
                                error = forecast - actual
                                individual_errors.append(error)
                                individual_abs_errors.append(abs(error))
                                individual_se.append(error ** 2)
                        
                        if individual_errors:
                            try:
                                result[f'{model_name}_ERROR'] = np.mean(individual_errors)
                                result[f'{model_name}_ABS_ERROR'] = np.mean(individual_abs_errors)
                                result[f'{model_name}_SE'] = np.mean(individual_se)
                            except Exception as e:
                                print(f"Error setting metrics for {model_name}: {e}")
                                print(f"Result type: {type(result)}")
                                print(f"Result content: {result}")
                                raise
            
            # Final check that result is still a dictionary
            if not isinstance(result, dict):
                raise TypeError(f"Result should be a dictionary before converting to Series, got {type(result)}")
            
            return pd.Series(result)
        
        # Group by week, location, department and aggregate AM/PM shifts
        filtered_df = filtered_df.groupby(['WEEK_NUMBER', 'WORK_LOCATION', 'DEPARTMENT_GROUP']).apply(aggregate_shifts).reset_index(drop=True)
        
        st.success(f"✅ Aggregated {len(filtered_df)} week/location/department combinations")
        
    else:
        # Specific shift(s) selected - show individual shift data
        if selected_shifts:
            filtered_df = filtered_df[filtered_df['SHIFT_TIME'].isin(selected_shifts)]
            st.info(f"📊 Showing individual shift data for: {', '.join(selected_shifts)}")
    
    # Calculate performance metrics after aggregation/filtering
    filtered_df = calculate_performance_metrics(filtered_df)
    
    # Apply performance filter based on best available model
    if filters['performance'] != "All Data":
        # Find the best performing model for each row to determine performance level
        available_models = ['GREYKITE', 'MA_4WEEK', 'MA_6WEEK', 'EXP_SMOOTH_02', 'EXP_SMOOTH_04', 'EXP_SMOOTH_06', 'EXP_SMOOTH_08', 'EXP_SMOOTH_10']
        ape_cols = [f'{model}_APE' for model in available_models if f'{model}_APE' in filtered_df.columns]
        
        if ape_cols:
            # Get minimum MAPE for each row across all models
            filtered_df['MIN_MAPE'] = filtered_df[ape_cols].min(axis=1)
            
            if filters['performance'] == "High Performance (MAPE < 3%)":
                filtered_df = filtered_df[filtered_df['MIN_MAPE'] < 3]
            elif filters['performance'] == "Medium Performance (3% ≤ MAPE < 6%)":
                filtered_df = filtered_df[(filtered_df['MIN_MAPE'] >= 3) & (filtered_df['MIN_MAPE'] < 6)]
            elif filters['performance'] == "Low Performance (MAPE ≥ 6%)":
                filtered_df = filtered_df[filtered_df['MIN_MAPE'] >= 6]
            
            # Remove the temporary column
            filtered_df = filtered_df.drop('MIN_MAPE', axis=1)
    
    # Final safety check before returning
    if not isinstance(filtered_df, pd.DataFrame):
        raise TypeError(f"apply_filters should return a DataFrame, got {type(filtered_df)}")
    
    # Ensure essential columns exist
    essential_columns = ['WEEK_NUMBER', 'WORK_LOCATION', 'DEPARTMENT_GROUP', 'ACTUAL_ATTENDANCE_RATE']
    missing_columns = [col for col in essential_columns if col not in filtered_df.columns]
    if missing_columns:
        raise KeyError(f"Essential columns missing from filtered DataFrame: {missing_columns}")
    
    return filtered_df

def show_data_info(df, filtered_df):
    """Display enhanced data information in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 DATASET OVERVIEW")
    
    # Original data info
    st.sidebar.markdown("**📈 Original Dataset:**")
    # Build info string conditionally
    info_parts = [
        f"• **Total Records**: {len(df):,}",
        f"• **Locations**: {df['WORK_LOCATION'].nunique()}",
        f"• **Departments**: {df['DEPARTMENT_GROUP'].nunique()}"
    ]
    
    # Add shifts info if column exists
    if 'SHIFT_TIME' in df.columns:
        info_parts.append(f"• **Shifts**: {df['SHIFT_TIME'].nunique()}")
    else:
        info_parts.append("• **Shifts**: Aggregated (AM/PM combined)")
    
    info_parts.append(f"• **Week Range**: {df['WEEK_NUMBER'].min()} to {df['WEEK_NUMBER'].max()}")
    
    st.sidebar.markdown("\n".join(info_parts) + "  ")
    
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
    """Create enhanced multi-model comparison KPI metrics dashboard (up to 8 models)."""
    st.markdown("### 📊 MODEL PERFORMANCE COMPARISON")
    st.markdown('<div class="kpi-section">', unsafe_allow_html=True)
    
    if len(df) == 0:
        st.warning("⚠️ No data available for the selected filters.")
        return
    
    # Define models and their display names (including all exponential smoothing variants)
    models = {
        'GREYKITE': 'Greykite',
        'MA_4WEEK': '4-Week MA',
        'MA_6WEEK': '6-Week MA',
        'EXP_SMOOTH_02': 'Exp. Smooth α=0.2',
        'EXP_SMOOTH_04': 'Exp. Smooth α=0.4',
        'EXP_SMOOTH_06': 'Exp. Smooth α=0.6',
        'EXP_SMOOTH_08': 'Exp. Smooth α=0.8',
        'EXP_SMOOTH_10': 'Exp. Smooth α=1.0'
    }
    
    # Calculate metrics for each available model
    metrics = {}
    total_weeks = len(df)
    
    for model_code, model_name in models.items():
        if f'{model_code}_APE' in df.columns:
            mape = df[f'{model_code}_APE'].mean()
            rmse = np.sqrt(df[f'{model_code}_SE'].mean())
            win_rate = (df[f'{model_code}_WINS'].sum() / total_weeks) * 100 if f'{model_code}_WINS' in df.columns else 0
            
            metrics[model_code] = {
                'name': model_name,
                'MAPE': mape,
                'RMSE': rmse,
                'WIN_RATE': win_rate,
                'WINS': df[f'{model_code}_WINS'].sum() if f'{model_code}_WINS' in df.columns else 0
            }
    
    if not metrics:
        st.warning("⚠️ No model performance data available.")
        return
    
    # Display metrics in columns based on available models
    available_models = list(metrics.keys())
    
    # Handle display for many models - use multiple rows if needed
    if len(available_models) <= 4:
        cols = st.columns(len(available_models))
        col_groups = [cols]
    else:
        # Split into rows of 4 columns each
        col_groups = []
        for i in range(0, len(available_models), 4):
            batch = available_models[i:i+4]
            cols = st.columns(len(batch))
            col_groups.append(cols)
    
    # Color scheme for different models (extended for more models)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    model_items = list(metrics.items())
    for i, (model_code, model_data) in enumerate(model_items):
        # Determine which column group and position
        group_idx = i // 4
        col_idx = i % 4
        
        if group_idx < len(col_groups) and col_idx < len(col_groups[group_idx]):
            with col_groups[group_idx][col_idx]:
                st.metric(
                    f"🎯 {model_data['name']}",
                    f"{model_data['MAPE']:.2f}%",
                    delta=f"Wins: {model_data['WINS']}/{total_weeks} ({model_data['WIN_RATE']:.1f}%)",
                    help=f"RMSE: {model_data['RMSE']:.2f}% | Individual week MAPE values averaged"
                )
    
    # Overall ranking section
    st.markdown("### 🏆 MODEL RANKING")
    
    # Rank by MAPE (lower is better)
    ranking = sorted(metrics.items(), key=lambda x: x[1]['MAPE'])
    
    rank_col1, rank_col2 = st.columns(2)
    
    with rank_col1:
        st.markdown("**📈 By Avg Weekly MAPE (Lower = Better)**")
        for i, (model_code, data) in enumerate(ranking):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "🔸"
            st.markdown(f"{medal} **{i+1}. {data['name']}** - {data['MAPE']:.2f}%")
    
    with rank_col2:
        st.markdown("**🏆 By Win Rate (Higher = Better)**")
        win_ranking = sorted(metrics.items(), key=lambda x: x[1]['WIN_RATE'], reverse=True)
        for i, (model_code, data) in enumerate(win_ranking):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "🔸"
            st.markdown(f"{medal} **{i+1}. {data['name']}** - {data['WIN_RATE']:.1f}%")
    
    # Performance insights
    if len(metrics) >= 2:
        best_model = ranking[0]
        worst_model = ranking[-1]
        
        improvement = ((worst_model[1]['MAPE'] - best_model[1]['MAPE']) / worst_model[1]['MAPE']) * 100
        
        st.markdown("### 📊 KEY INSIGHTS")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🏆 Best Model",
                best_model[1]['name'],
                delta=f"{best_model[1]['MAPE']:.2f}% Avg Weekly MAPE"
            )
        
        with col2:
            st.metric(
                "📈 Performance Gap",
                f"{improvement:.1f}%",
                delta=f"{worst_model[1]['MAPE'] - best_model[1]['MAPE']:.2f}pp"
            )
        
        with col3:
            total_wins = sum(data['WINS'] for data in metrics.values())
            st.metric(
                "📊 Total Comparisons",
                f"{total_weeks}",
                delta=f"{total_wins} decisive wins"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_outliers_table(df):
    """Create comprehensive outliers analysis for all available models."""
    st.markdown("### 🚨 OUTLIERS ANALYSIS - ALL MODELS")
    
    if len(df) == 0:
        st.warning("⚠️ No data available for the selected filters.")
        return
    
    # Define models and their forecast columns (including all exponential smoothing variants)
    models_config = {
        'GREYKITE': {'name': 'Greykite', 'forecast_col': 'GREYKITE_FORECAST'},
        'MA_4WEEK': {'name': '4-Week MA', 'forecast_col': 'MOVING_AVG_4WEEK_FORECAST'},
        'MA_6WEEK': {'name': '6-Week MA', 'forecast_col': 'SIX_WEEK_ROLLING_AVG'},
        'EXP_SMOOTH_02': {'name': 'Exp. Smooth α=0.2', 'forecast_col': 'EXP_SMOOTH_02'},
        'EXP_SMOOTH_04': {'name': 'Exp. Smooth α=0.4', 'forecast_col': 'EXP_SMOOTH_04'},
        'EXP_SMOOTH_06': {'name': 'Exp. Smooth α=0.6', 'forecast_col': 'EXP_SMOOTH_06'},
        'EXP_SMOOTH_08': {'name': 'Exp. Smooth α=0.8', 'forecast_col': 'EXP_SMOOTH_08'},
        'EXP_SMOOTH_10': {'name': 'Exp. Smooth α=1.0', 'forecast_col': 'EXP_SMOOTH_10'}
    }
    
    # Get outliers for any available model
    outlier_conditions = []
    available_models = []
    
    for model_code, config in models_config.items():
        outlier_col = f'{model_code}_APE_OUTLIER'
        if outlier_col in df.columns:
            outlier_conditions.append(df[outlier_col] == 1)
            available_models.append(model_code)
    
    if not outlier_conditions:
        st.warning("⚠️ No outlier data available for analysis.")
        return
    
    # Combine all outlier conditions
    outliers_mask = pd.concat(outlier_conditions, axis=1).any(axis=1)
    outliers_df = df[outliers_mask].copy()
    
    if len(outliers_df) == 0:
        st.info("✅ No outliers detected in the current filtered dataset.")
        return
    
    # Summary metrics by model
    st.markdown("### 📊 OUTLIER SUMMARY BY MODEL")
    
    # Handle display for many models - use multiple rows if needed
    if len(available_models) <= 4:
        cols = st.columns(len(available_models))
        col_groups = [cols]
    else:
        # Split into rows of 4 columns each
        col_groups = []
        for i in range(0, len(available_models), 4):
            batch = available_models[i:i+4]
            cols = st.columns(len(batch))
            col_groups.append(cols)
    
    total_outliers_by_model = {}
    
    for i, model_code in enumerate(available_models):
        config = models_config[model_code]
        outlier_col = f'{model_code}_APE_OUTLIER'
        outlier_count = df[outlier_col].sum()
        total_outliers_by_model[model_code] = outlier_count
        
        # Determine which column group and position
        group_idx = i // 4
        col_idx = i % 4
        
        if group_idx < len(col_groups) and col_idx < len(col_groups[group_idx]):
            with col_groups[group_idx][col_idx]:
                st.metric(
                    f"🚨 {config['name']}",
                    f"{outlier_count} outliers",
                    delta=f"{(outlier_count/len(df)*100):.1f}% of data"
                )
    
    # Detailed outliers table
    st.markdown("### 📋 DETAILED OUTLIERS INFORMATION")
    
    # Prepare display columns
    base_columns = ['WEEK_NUMBER', 'WORK_LOCATION', 'DEPARTMENT_GROUP', 'ACTUAL_ATTENDANCE_RATE']
    
    # Add SHIFT_TIME column if it exists (not aggregated)
    if 'SHIFT_TIME' in df.columns:
        base_columns.insert(3, 'SHIFT_TIME')
    
    # Add forecast and MAPE columns for available models
    display_columns = base_columns.copy()
    for model_code in available_models:
        config = models_config[model_code]
        forecast_col = config['forecast_col']
        mape_col = f'{model_code}_APE'
        
        if forecast_col in df.columns:
            display_columns.extend([forecast_col, mape_col])
    
    # Create outlier type indicator
    outliers_display = outliers_df[display_columns].copy()
    
    # Add outlier type column
    def get_outlier_type(row):
        outlier_models = []
        for model_code in available_models:
            outlier_col = f'{model_code}_APE_OUTLIER'
            if outlier_col in row and row[outlier_col] == 1:
                outlier_models.append(models_config[model_code]['name'])
        return ', '.join(outlier_models) if outlier_models else 'None'
    
    outliers_display['Outlier_Models'] = outliers_df.apply(get_outlier_type, axis=1)
    
    # Format numerical columns
    for col in outliers_display.columns:
        if col in ['ACTUAL_ATTENDANCE_RATE'] or any(forecast_col in col for forecast_col in [config['forecast_col'] for config in models_config.values()]):
            outliers_display[col] = outliers_display[col].round(2)
        elif '_APE' in col:
            outliers_display[col] = outliers_display[col].round(2)
    
    # Rename columns for better display
    column_renames = {
        'WEEK_NUMBER': 'Week',
        'WORK_LOCATION': 'Location',
        'DEPARTMENT_GROUP': 'Department',
        'ACTUAL_ATTENDANCE_RATE': 'Actual (%)',
        'Outlier_Models': 'Outlier in Models'
    }
    
    # Add SHIFT_TIME rename if column exists
    if 'SHIFT_TIME' in outliers_display.columns:
        column_renames['SHIFT_TIME'] = 'Shift'
    
    for model_code in available_models:
        config = models_config[model_code]
        forecast_col = config['forecast_col']
        mape_col = f'{model_code}_APE'
        
        if forecast_col in outliers_display.columns:
            column_renames[forecast_col] = f"{config['name']} Forecast (%)"
        if mape_col in outliers_display.columns:
            column_renames[mape_col] = f"{config['name']} Weekly MAPE (%)"
    
    outliers_display = outliers_display.rename(columns=column_renames)
    
    # Display the table
    st.dataframe(
        outliers_display.sort_values(['Week', 'Location']),
        use_container_width=True,
        height=400
    )
    
    # Outlier analysis insights
    st.markdown("### 🔍 OUTLIER INSIGHTS")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Outlier Distribution by Model**")
        
        # Create a simple bar chart of outlier counts
        outlier_counts = []
        model_names = []
        
        for model_code in available_models:
            config = models_config[model_code]
            count = total_outliers_by_model[model_code]
            outlier_counts.append(count)
            model_names.append(config['name'])
        
        outlier_fig = go.Figure(data=[
            go.Bar(
                x=model_names,
                y=outlier_counts,
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(model_names)]
            )
        ])
        
        outlier_fig.update_layout(
            title="Outlier Count by Model",
            xaxis_title="Model",
            yaxis_title="Number of Outliers",
            height=300
        )
        
        st.plotly_chart(outlier_fig, use_container_width=True)
    
    with col2:
        st.markdown("**🎯 Worst Performing Cases**")
        
        # Find worst case for each model
        for model_code in available_models:
            config = models_config[model_code]
            mape_col = f'{model_code}_APE'
            
            if mape_col in outliers_df.columns and len(outliers_df) > 0:
                worst_case = outliers_df.loc[outliers_df[mape_col].idxmax()]
                st.markdown(f"""
                **{config['name']}:**
                - Week: {worst_case.get('WEEK_NUMBER', 'N/A')}
                - Location: {worst_case.get('WORK_LOCATION', 'N/A')}
                - Weekly MAPE: {worst_case[mape_col]:.2f}%
                """)
    
    # Common outlier patterns
    if len(outliers_df) > 0:
        st.markdown("### 📈 COMMON OUTLIER PATTERNS")
        
        pattern_col1, pattern_col2, pattern_col3 = st.columns(3)
        
        with pattern_col1:
            st.markdown("**🏢 Most Problematic Locations**")
            location_outliers = outliers_df['WORK_LOCATION'].value_counts().head(3)
            for location, count in location_outliers.items():
                st.markdown(f"• {location}: {count} outliers")
        
        with pattern_col2:
            st.markdown("**🏭 Most Problematic Departments**")
            dept_outliers = outliers_df['DEPARTMENT_GROUP'].value_counts().head(3)
            for dept, count in dept_outliers.items():
                st.markdown(f"• {dept}: {count} outliers")
        
        with pattern_col3:
            if 'SHIFT_TIME' in outliers_df.columns:
                st.markdown("**⏰ Most Problematic Shifts**")
                shift_outliers = outliers_df['SHIFT_TIME'].value_counts().head(3)
                for shift, count in shift_outliers.items():
                    st.markdown(f"• {shift}: {count} outliers")
            else:
                st.markdown("**📊 Data Note**")
                st.markdown("• Shift analysis not available")
                st.markdown("• AM/PM shifts aggregated")
                st.markdown("• Focus on location/department patterns")

def create_weekly_mape_trends(df):
    """Create weekly MAPE trends chart for all available models."""
    st.markdown("### 📅 WEEKLY MAPE TRENDS - ALL MODELS")
    st.markdown("*Note: Each point represents the average weekly MAPE for that specific week across all locations/departments*")
    
    if len(df) == 0:
        st.warning("⚠️ No data available for the selected filters.")
        return
    
    # Define models and their properties (including all exponential smoothing variants)
    models_config = {
        'GREYKITE': {'name': 'Greykite', 'color': '#1f77b4'},
        'MA_4WEEK': {'name': '4-Week MA', 'color': '#ff7f0e'},
        'MA_6WEEK': {'name': '6-Week MA', 'color': '#2ca02c'},
        'EXP_SMOOTH_02': {'name': 'Exp. Smooth α=0.2', 'color': '#d62728'},
        'EXP_SMOOTH_04': {'name': 'Exp. Smooth α=0.4', 'color': '#9467bd'},
        'EXP_SMOOTH_06': {'name': 'Exp. Smooth α=0.6', 'color': '#8c564b'},
        'EXP_SMOOTH_08': {'name': 'Exp. Smooth α=0.8', 'color': '#e377c2'},
        'EXP_SMOOTH_10': {'name': 'Exp. Smooth α=1.0', 'color': '#7f7f7f'}
    }
    
    # Prepare weekly aggregated data for all available models
    agg_dict = {}
    available_models = []
    
    for model_code, config in models_config.items():
        if f'{model_code}_APE' in df.columns:
            agg_dict[f'{model_code}_APE'] = 'mean'
            available_models.append(model_code)
    
    if not agg_dict:
        st.warning("⚠️ No model data available for trend analysis.")
        return
    
    # Add win rate data if available
    if 'BEST_MODEL' in df.columns:
        for model_code in available_models:
            wins_col = f'{model_code}_WINS'
            if wins_col in df.columns:
                agg_dict[wins_col] = 'mean'
    
    weekly_data = df.groupby('WEEK_NUMBER').agg(agg_dict).reset_index()
    
    # Rename columns to match expected names
    rename_dict = {}
    for model_code in available_models:
        rename_dict[f'{model_code}_APE'] = f'{model_code}_MAPE'
        wins_col = f'{model_code}_WINS'
        if wins_col in weekly_data.columns:
            rename_dict[wins_col] = f'{model_code}_WIN_RATE'
    
    weekly_data = weekly_data.rename(columns=rename_dict)
    
    # Create the main trends chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Weekly MAPE Comparison - All Models", "Weekly Win Rate Distribution"),
        vertical_spacing=0.15,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # MAPE trends for all models
    for model_code in available_models:
        config = models_config[model_code]
        mape_col = f'{model_code}_MAPE'
        
        if mape_col in weekly_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=weekly_data['WEEK_NUMBER'],
                    y=weekly_data[mape_col],
                    mode='lines+markers',
                    name=config['name'],
                    line=dict(color=config['color'], width=3),
                    marker=dict(size=8),
                    hovertemplate=f"<b>{config['name']}</b><br>Week: %{{x}}<br>Weekly MAPE: %{{y:.2f}}%<extra></extra>"
                ),
                row=1, col=1
            )
    
    # Win rate stacked area chart
    if 'BEST_MODEL' in df.columns:
        for model_code in available_models:
            config = models_config[model_code]
            win_rate_col = f'{model_code}_WIN_RATE'
            
            if win_rate_col in weekly_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=weekly_data['WEEK_NUMBER'],
                        y=weekly_data[win_rate_col] * 100,
                        mode='lines+markers',
                        name=f"{config['name']} Win Rate",
                        line=dict(color=config['color'], width=2),
                        marker=dict(size=6),
                        fill='tonexty' if model_code != available_models[0] else None,
                        hovertemplate=f"<b>{config['name']}</b><br>Week: %{{x}}<br>Win Rate: %{{y:.1f}}%<extra></extra>",
                        showlegend=False
                    ),
                    row=2, col=1
                )
    
    # Update layout
    fig.update_xaxes(title_text="Week", row=2, col=1)
    fig.update_yaxes(title_text="Weekly MAPE (%)", row=1, col=1)
    fig.update_yaxes(title_text="Win Rate (%)", row=2, col=1)
    
    fig.update_layout(
        height=700,
        title_text="📊 COMPREHENSIVE MODEL PERFORMANCE TRENDS",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.markdown("### 📊 TREND SUMMARY STATISTICS")
    
    summary_cols = st.columns(len(available_models))
    
    for i, model_code in enumerate(available_models):
        config = models_config[model_code]
        mape_col = f'{model_code}_MAPE'
        
        if mape_col in weekly_data.columns:
            avg_mape = weekly_data[mape_col].mean()
            std_mape = weekly_data[mape_col].std()
            min_mape = weekly_data[mape_col].min()
            max_mape = weekly_data[mape_col].max()
            
            with summary_cols[i]:
                st.markdown(f"""
                **🎯 {config['name']}**
                - Avg Weekly MAPE: {avg_mape:.2f}%
                - Std Dev: {std_mape:.2f}%
                - Range: {min_mape:.2f}% - {max_mape:.2f}%
                """)
    
    # Volatility analysis
    if len(available_models) >= 2:
        st.markdown("### 📈 MODEL STABILITY ANALYSIS")
        
        stability_data = []
        for model_code in available_models:
            config = models_config[model_code]
            mape_col = f'{model_code}_MAPE'
            
            if mape_col in weekly_data.columns:
                volatility = weekly_data[mape_col].std()
                stability_data.append({
                    'Model': config['name'],
                    'Volatility (Std Dev)': volatility,
                    'Stability Score': 1 / (1 + volatility)  # Higher score = more stable
                })
        
        stability_df = pd.DataFrame(stability_data)
        stability_df = stability_df.sort_values('Stability Score', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏆 Most Stable Model**")
            most_stable = stability_df.iloc[0]
            st.success(f"**{most_stable['Model']}** (Volatility: {most_stable['Volatility (Std Dev)']:.2f}%)")
        
        with col2:
            st.markdown("**📊 Stability Ranking**")
            for i, row in stability_df.iterrows():
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "🔸"
                st.markdown(f"{medal} {row['Model']}: {row['Volatility (Std Dev)']:.2f}%")

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

def create_model_comparison_matrix(df):
    """Create a comprehensive model comparison matrix."""
    st.markdown("### 🔍 MODEL COMPARISON MATRIX")
    
    if len(df) == 0:
        st.warning("⚠️ No data available for the selected filters.")
        return
    
    # Define models
    models = {
        'GREYKITE': 'Greykite',
        'MA_4WEEK': '4-Week MA',
        'MA_6WEEK': '6-Week MA',
        'EXP_SMOOTH': 'Exp. Smoothing'
    }
    
    # Calculate correlation matrix between model forecasts
    available_models = []
    forecast_cols = []
    
    for model_code, model_name in models.items():
        forecast_col = f'{model_code}_FORECAST' if model_code == 'GREYKITE' else \
                      'MOVING_AVG_4WEEK_FORECAST' if model_code == 'MA_4WEEK' else \
                      'SIX_WEEK_ROLLING_AVG' if model_code == 'MA_6WEEK' else \
                      'EXPONENTIAL_SMOOTHING'
        
        if forecast_col in df.columns:
            available_models.append(model_name)
            forecast_cols.append(forecast_col)
    
    if len(forecast_cols) >= 2:
        # Create correlation matrix
        corr_matrix = df[forecast_cols].corr()
        corr_matrix.index = available_models
        corr_matrix.columns = available_models
        
        # Display correlation heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Model Forecast Correlation Matrix",
            color_continuous_scale="RdBu_r"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance comparison table
        st.markdown("### 📊 DETAILED PERFORMANCE METRICS")
        
        performance_data = []
        for i, model_code in enumerate(['GREYKITE', 'MA_4WEEK', 'MA_6WEEK', 'EXP_SMOOTH']):
            if f'{model_code}_APE' in df.columns:
                model_name = models[model_code]
                mape = df[f'{model_code}_APE'].mean()
                median_ape = df[f'{model_code}_APE'].median()
                rmse = np.sqrt(df[f'{model_code}_SE'].mean())
                win_rate = (df[f'{model_code}_WINS'].sum() / len(df)) * 100 if f'{model_code}_WINS' in df.columns else 0
                
                performance_data.append({
                    'Model': model_name,
                    'MAPE (%)': round(mape, 2),
                    'Median APE (%)': round(median_ape, 2),
                    'RMSE (%)': round(rmse, 2),
                    'Win Rate (%)': round(win_rate, 1),
                    'Wins': df[f'{model_code}_WINS'].sum() if f'{model_code}_WINS' in df.columns else 0
                })
        
        if performance_data:
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True)

def create_pairwise_comparison(df, model1_code, model2_code):
    """Create detailed pairwise comparison between two models."""
    models = {
        'GREYKITE': 'Greykite',
        'MA_4WEEK': '4-Week MA',
        'MA_6WEEK': '6-Week MA',
        'EXP_SMOOTH': 'Exp. Smoothing'
    }
    
    model1_name = models.get(model1_code, model1_code)
    model2_name = models.get(model2_code, model2_code)
    
    st.markdown(f"### ⚔️ {model1_name} vs {model2_name} COMPARISON")
    
    # Check if both models have data
    if f'{model1_code}_APE' not in df.columns or f'{model2_code}_APE' not in df.columns:
        st.warning("⚠️ One or both models don't have data available.")
        return
    
    # Performance metrics comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model1_mape = df[f'{model1_code}_APE'].mean()
        model2_mape = df[f'{model2_code}_APE'].mean()
        winner = model1_name if model1_mape < model2_mape else model2_name
        st.metric(
            "🏆 MAPE Winner",
            winner,
            delta=f"{abs(model1_mape - model2_mape):.2f}pp difference"
        )
    
    with col2:
        model1_wins = df[f'{model1_code}_WINS'].sum() if f'{model1_code}_WINS' in df.columns else 0
        model2_wins = df[f'{model2_code}_WINS'].sum() if f'{model2_code}_WINS' in df.columns else 0
        st.metric(
            f"🎯 {model1_name} Wins",
            f"{model1_wins}/{len(df)}",
            delta=f"{(model1_wins/len(df)*100):.1f}%"
        )
    
    with col3:
        st.metric(
            f"📊 {model2_name} Wins", 
            f"{model2_wins}/{len(df)}",
            delta=f"{(model2_wins/len(df)*100):.1f}%"
        )
    
    # Scatter plot comparison
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df[f'{model1_code}_APE'],
        y=df[f'{model2_code}_APE'],
        mode='markers',
        name='Data Points',
        hovertemplate=f'<b>{model1_name} Weekly MAPE:</b> %{{x:.2f}}%<br><b>{model2_name} Weekly MAPE:</b> %{{y:.2f}}%<extra></extra>'
    ))
    
    # Add diagonal line (equal performance)
    max_val = max(df[f'{model1_code}_APE'].max(), df[f'{model2_code}_APE'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Equal Performance',
        line=dict(dash='dash', color='red')
    ))
    
    fig.update_layout(
        title=f"{model1_name} vs {model2_name} Weekly MAPE Comparison",
        xaxis_title=f"{model1_name} Weekly MAPE (%)",
        yaxis_title=f"{model2_name} Weekly MAPE (%)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_best_worst_analysis(df):
    """Analyze best vs worst performing model scenarios."""
    st.markdown("### 🏆 BEST vs WORST PERFORMANCE ANALYSIS")
    
    if 'BEST_MODEL' not in df.columns:
        st.warning("⚠️ Best model data not available.")
        return
    
    # Best model distribution
    best_model_counts = df['BEST_MODEL'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏆 Best Model Distribution**")
        fig = px.pie(
            values=best_model_counts.values,
            names=best_model_counts.index,
            title="Which Model Wins Most Often?"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**📊 Performance by Scenario**")
        
        # Find scenarios where each model performs best
        models = ['GREYKITE', 'MA_4WEEK', 'MA_6WEEK', 'EXP_SMOOTH']
        scenario_analysis = []
        
        for model in models:
            if model in df['BEST_MODEL'].values:
                model_best_df = df[df['BEST_MODEL'] == model]
                avg_mape = model_best_df[f'{model}_APE'].mean() if f'{model}_APE' in df.columns else 0
                scenario_analysis.append({
                    'Model': model,
                    'Wins': len(model_best_df),
                    'Avg MAPE When Best': round(avg_mape, 2)
                })
        
        if scenario_analysis:
            scenario_df = pd.DataFrame(scenario_analysis)
            st.dataframe(scenario_df, use_container_width=True)

def create_individual_model_analysis(df, model_code, model_name):
    """Create detailed analysis for a single model."""
    st.markdown(f"### 🔍 {model_name} DETAILED ANALYSIS")
    
    if f'{model_code}_APE' not in df.columns:
        st.warning(f"⚠️ {model_name} data not available.")
        return
    
    # Performance metrics
    mape = df[f'{model_code}_APE'].mean()
    median_ape = df[f'{model_code}_APE'].median()
    std_ape = df[f'{model_code}_APE'].std()
    wins = df[f'{model_code}_WINS'].sum() if f'{model_code}_WINS' in df.columns else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Average Weekly MAPE", f"{mape:.2f}%")
    with col2:
        st.metric("📈 Median Weekly MAPE", f"{median_ape:.2f}%")
    with col3:
        st.metric("📏 Std Deviation", f"{std_ape:.2f}%")
    with col4:
        st.metric("🏆 Total Wins", f"{wins}/{len(df)}")
    
    # Distribution plot
    fig = px.histogram(
        df,
        x=f'{model_code}_APE',
        nbins=30,
        title=f"{model_name} Weekly MAPE Distribution",
        labels={f'{model_code}_APE': 'Weekly MAPE (%)'}
    )
    
    fig.add_vline(x=mape, line_dash="dash", line_color="red", annotation_text=f"Mean: {mape:.2f}%")
    fig.add_vline(x=median_ape, line_dash="dash", line_color="green", annotation_text=f"Median: {median_ape:.2f}%")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance by dimensions
    st.markdown(f"### 📊 {model_name} PERFORMANCE BY DIMENSIONS")
    
    dimension_col1, dimension_col2 = st.columns(2)
    
    with dimension_col1:
        # Performance by location
        location_perf = df.groupby('WORK_LOCATION')[f'{model_code}_APE'].mean().sort_values()
        
        fig = px.bar(
            x=location_perf.values,
            y=location_perf.index,
            orientation='h',
            title=f"{model_name} Weekly MAPE by Location",
            labels={'x': 'Average Weekly MAPE (%)', 'y': 'Location'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with dimension_col2:
        # Performance by department
        dept_perf = df.groupby('DEPARTMENT_GROUP')[f'{model_code}_APE'].mean().sort_values()
        
        fig = px.bar(
            x=dept_perf.values,
            y=dept_perf.index,
            orientation='h',
            title=f"{model_name} Weekly MAPE by Department",
            labels={'x': 'Average Weekly MAPE (%)', 'y': 'Department'}
        )
        st.plotly_chart(fig, use_container_width=True)

def configure_page():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="Attendance Forecasting Analytics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced CSS styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header h3 {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .kpi-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #7f8c8d;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-top: 2rem;
    }
    
    .stMetric {
        background-color: #ffffff !important;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e1e5e9;
    }
    
    .stMetric [data-testid="metric-container"] {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    
    .stMetric [data-testid="metric-value"] {
        color: #262730 !important;
        font-weight: 600;
    }
    
    .stMetric [data-testid="metric-label"] {
        color: #525252 !important;
        font-weight: 500;
    }
    
    .stMetric [data-testid="metric-delta"] {
        color: #28a745 !important;
        font-weight: 500;
    }
    
    .stMetric [data-testid="metric-delta"][data-color="negative"] {
        color: #dc3545 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main dashboard application with enhanced 4-model comparison functionality."""
    # Configure page
    configure_page()
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Create sidebar filters
    filters = create_filters(df)
    
    # Apply filters
    filtered_df = apply_filters(df, filters)
    
    # Debug information for troubleshooting
    st.sidebar.markdown("### 🔧 Debug Info")
    st.sidebar.markdown(f"**Filtered DF Shape:** {filtered_df.shape}")
    st.sidebar.markdown(f"**Filtered DF Type:** {type(filtered_df)}")
    if hasattr(filtered_df, 'columns'):
        st.sidebar.markdown(f"**Available Columns:** {len(filtered_df.columns)}")
        with st.sidebar.expander("View All Columns"):
            st.write(list(filtered_df.columns))
    else:
        st.sidebar.error("⚠️ Filtered DataFrame has no columns attribute!")
    
    # Main dashboard header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 ATTENDANCE FORECASTING ANALYTICS DASHBOARD</h1>
        <h3>📊 Enhanced Multi-Model Comparison: Greykite with Confidence Intervals | Multiple Exponential Smoothing Variants | Moving Averages</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display data summary
    if len(filtered_df) > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Records", f"{len(filtered_df):,}")
        with col2:
            # Safe access to columns with error handling
            if 'WORK_LOCATION' in filtered_df.columns:
                unique_locations = filtered_df['WORK_LOCATION'].nunique()
                st.metric("🏢 Locations", unique_locations)
            else:
                st.metric("🏢 Locations", "N/A")
                st.error(f"Missing WORK_LOCATION column. Available columns: {list(filtered_df.columns)}")
        with col3:
            if 'DEPARTMENT_GROUP' in filtered_df.columns:
                unique_departments = filtered_df['DEPARTMENT_GROUP'].nunique()
                st.metric("🏭 Departments", unique_departments)
            else:
                st.metric("🏭 Departments", "N/A")
                st.error(f"Missing DEPARTMENT_GROUP column. Available columns: {list(filtered_df.columns)}")
        with col4:
            if 'WEEK_NUMBER' in filtered_df.columns:
                week_count = len(filtered_df['WEEK_NUMBER'].unique())
                st.metric("📅 Weeks", f"{week_count} weeks")
            else:
                st.metric("📅 Weeks", "N/A")
                st.error(f"Missing WEEK_NUMBER column. Available columns: {list(filtered_df.columns)}")
    
    # Create visualizations based on selected comparison type
    comparison_type = filters['comparison']
    
    if comparison_type == 'All Models Overview':
        # Show comprehensive overview of all models
        create_enhanced_kpi_metrics(filtered_df)
        
        # Model performance comparison charts
        st.markdown("---")
        create_weekly_mape_trends(filtered_df)
        
        # Outliers analysis
        st.markdown("---")
        create_outliers_table(filtered_df)
        
        # Additional comprehensive analysis
        create_model_comparison_matrix(filtered_df)
        
    elif comparison_type == 'Greykite vs 4-Week MA':
        # Legacy comparison - focus on these two models
        create_pairwise_comparison(filtered_df, 'GREYKITE', 'MA_4WEEK')
        
    elif comparison_type == 'Greykite vs 6-Week MA':
        create_pairwise_comparison(filtered_df, 'GREYKITE', 'MA_6WEEK')
        
    elif comparison_type == 'Greykite vs Exp. Smoothing':
        # Use the middle alpha value (0.6) as representative exponential smoothing
        if 'EXP_SMOOTH_06' in filtered_df.columns:
            create_pairwise_comparison(filtered_df, 'GREYKITE', 'EXP_SMOOTH_06')
        else:
            st.warning("⚠️ Exponential smoothing data not available in this dataset")
        
    elif comparison_type == 'Moving Average Comparison':
        create_pairwise_comparison(filtered_df, 'MA_4WEEK', 'MA_6WEEK')
        
    elif comparison_type == 'Exponential Smoothing Variants':
        # Compare different exponential smoothing alpha values
        if 'EXP_SMOOTH_02' in filtered_df.columns and 'EXP_SMOOTH_04' in filtered_df.columns:
            st.markdown("### 📊 Exponential Smoothing Alpha Comparison")
            st.info("Comparing different alpha values (smoothing parameters) for exponential smoothing")
            create_pairwise_comparison(filtered_df, 'EXP_SMOOTH_02', 'EXP_SMOOTH_04')
            st.markdown("---")
            create_pairwise_comparison(filtered_df, 'EXP_SMOOTH_04', 'EXP_SMOOTH_08')
        else:
            st.warning("⚠️ Exponential smoothing variants not available in this dataset")
        
    elif comparison_type == 'Best vs Worst Performance':
        create_best_worst_analysis(filtered_df)
        
    elif comparison_type == 'Individual Model Focus':
        focus_model = filters['focus_model']
        if focus_model:
            model_mapping = {
                'Greykite': 'GREYKITE',
                '4-Week MA': 'MA_4WEEK', 
                '6-Week MA': 'MA_6WEEK',
                'Exp. Smooth α=0.2': 'EXP_SMOOTH_02',
                'Exp. Smooth α=0.4': 'EXP_SMOOTH_04',
                'Exp. Smooth α=0.6': 'EXP_SMOOTH_06',
                'Exp. Smooth α=0.8': 'EXP_SMOOTH_08',
                'Exp. Smooth α=1.0': 'EXP_SMOOTH_10'
            }
            model_code = model_mapping.get(focus_model)
            if model_code:
                create_individual_model_analysis(filtered_df, model_code, focus_model)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>📊 <strong>Dashboard Features:</strong> Enhanced Greykite Forecasting | 5 Exponential Smoothing Variants (α=0.2,0.4,0.6,0.8,1.0) | Confidence Intervals | Performance Analytics | Outlier Detection | Trend Analysis</p>
        <p>🔄 <strong>Last Updated:</strong> Real-time data processing with automatic file detection and advanced filtering capabilities</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
