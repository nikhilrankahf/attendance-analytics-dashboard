#!/usr/bin/env python3
"""
Modified Attendance Greykite Integration Script
This script reads attendance data from a local CSV file and runs Greykite forecasting
with backtesting to compare actual vs forecasted attendance rates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import warnings
from collections import defaultdict
from sklearn.metrics import mean_absolute_error

# Greykite imports
from greykite.framework.templates.autogen.forecast_config import (
    EvaluationPeriodParam, ForecastConfig, MetadataParam, ModelComponentsParam,
    EvaluationMetricParam
)
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.utils.result_summary import summarize_grid_search_results

warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = (12, 6)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', None)

# Helper functions for HelloFresh week format
def datetime_to_hf_week(date_time: dt.datetime):
    """Serialize a Python datetime into a HelloFresh week - e.g. '2017-W06'"""
    hf_datetime = date_time + dt.timedelta(days=2)
    year, week, _ = hf_datetime.isocalendar()
    return f'{year}-W{week:02d}'

def hf_week_to_datetime(hf_week: str):
    """Parse a HelloFresh stringified date into a Python datetime"""
    return dt.datetime.strptime(hf_week + '-1', '%G-W%V-%w') - dt.timedelta(days=2)

def hf_week_add(hf_week: str, num_weeks: int):
    return datetime_to_hf_week(hf_week_to_datetime(hf_week) + dt.timedelta(weeks=num_weeks))

def hf_week_sub(hf_week: str, num_weeks: int):
    return datetime_to_hf_week(hf_week_to_datetime(hf_week) - dt.timedelta(weeks=num_weeks))

def current_hf_week():
    return datetime_to_hf_week(dt.datetime.today())

def hf_week_to_last_thursday_str(hf_week: str):
    """Return the Monday of a given HelloFresh week as a string in 'YYYY-MM-DD' format."""
    monday = hf_week_to_datetime(hf_week) - dt.timedelta(days=2)
    return monday.strftime('%Y-%m-%d')

def calculate_moving_average(df, window=4):
    """Calculate 4-week moving average for attendance rate"""
    return df['WEEKLY_ATTENDANCE_RATE'].rolling(window=window, min_periods=1).mean()

def run_greykite_backtest(df_group, actual_col='WEEKLY_ATTENDANCE_RATE', test_horizon=4):
    """
    Run Greykite model with backtesting
    Returns DataFrame with actual vs predicted values for test period
    """
    if len(df_group) < 8:  # Need minimum data points
        return None
    
    metadata = MetadataParam(
        time_col="WEEK_BEGIN",
        value_col=actual_col,
        freq="W-THU"
    )
    
    forecaster = Forecaster()
    
    try:
        result = forecaster.run_forecast_config(
            df=df_group,
            config=ForecastConfig(
                model_template=ModelTemplateEnum.AUTO.name,
                forecast_horizon=test_horizon,
                coverage=0.95,
                metadata_param=metadata,
                model_components_param=ModelComponentsParam(
                    events={
                        "holiday_lookup_countries": ["US"],
                        "holiday_pre_num_days": 8,
                    }
                ),
                evaluation_metric_param=EvaluationMetricParam(
                    cv_selection_metric="MeanAbsoluteError",
                ),
                evaluation_period_param=EvaluationPeriodParam(
                    test_horizon=test_horizon,
                    periods_between_train_test=0,
                    cv_horizon=test_horizon,
                    cv_min_train_periods=max(4, len(df_group) - test_horizon - 1)
                )
            )
        )
        
        # Extract backtest results
        backtest_df = result.backtest.df.copy()
        backtest_df = backtest_df[['ts', 'actual', 'forecast']].rename(columns={
            'ts': 'WEEK_BEGIN',
            'actual': 'ACTUAL_ATTENDANCE_RATE',
            'forecast': 'GREYKITE_FORECAST'
        })
        
        return backtest_df
        
    except Exception as e:
        print(f"Error in Greykite forecasting: {e}")
        return None

def main():
    """Main function to run the modified attendance forecasting script"""
    
    # Read data from local CSV file
    csv_path = "/Users/nikhil.ranka/Labor_Management_-_Greykite_Input.csv"
    
    try:
        df_attendance = pd.read_csv(csv_path)
        print(f"Successfully loaded data from {csv_path}")
        print(f"Data shape: {df_attendance.shape}")
        print(f"Columns: {list(df_attendance.columns)}")
    except FileNotFoundError:
        print(f"Error: Could not find file {csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Data preprocessing
    # Ensure WEEK_BEGIN is datetime
    if 'WEEK_BEGIN' in df_attendance.columns:
        df_attendance['WEEK_BEGIN'] = pd.to_datetime(df_attendance['WEEK_BEGIN'])
    else:
        print("Error: WEEK_BEGIN column not found in data")
        return
    
    # Filter out attendance rates over 130% (as per original logic)
    if 'WEEKLY_ATTENDANCE_RATE' in df_attendance.columns:
        df_attendance = df_attendance[df_attendance['WEEKLY_ATTENDANCE_RATE'] < 130]
    else:
        print("Error: WEEKLY_ATTENDANCE_RATE column not found in data")
        return
    
    # Remove duplicates
    key_columns = ['WEEK_NUMBER', 'WORK_LOCATION', 'SHIFT_TIME']
    if 'DEPARTMENT_GROUP' in df_attendance.columns:
        key_columns.append('DEPARTMENT_GROUP')
    
    available_key_columns = [col for col in key_columns if col in df_attendance.columns]
    df_attendance = df_attendance.drop_duplicates(subset=available_key_columns, keep='last')
    
    # Sort by week
    df_attendance = df_attendance.sort_values(['WEEK_BEGIN'])
    
    print(f"After preprocessing: {df_attendance.shape}")
    print(f"Date range: {df_attendance['WEEK_BEGIN'].min()} to {df_attendance['WEEK_BEGIN'].max()}")
    
    # Group columns for iteration
    group_columns = ['WORK_LOCATION', 'SHIFT_TIME']
    if 'DEPARTMENT_GROUP' in df_attendance.columns:
        group_columns.append('DEPARTMENT_GROUP')
    
    # Results storage
    all_results = []
    
    # Process each unique combination
    for group_values, df_group in df_attendance.groupby(group_columns):
        
        if len(group_columns) == 3:
            work_location, shift_time, department_group = group_values
        else:
            work_location, shift_time = group_values
            department_group = 'Unknown'
        
        print(f"\nProcessing: {work_location} - {shift_time} - {department_group}")
        print(f"Data points: {len(df_group)}")
        
        if len(df_group) < 8:  # Need minimum data for meaningful forecasting
            print(f"Skipping due to insufficient data points")
            continue
        
        # Reset index for the group
        df_group = df_group.reset_index(drop=True)
        
        # Calculate 4-week moving average
        
        # Run Greykite backtesting
        greykite_results = run_greykite_backtest(df_group)
        
        if greykite_results is not None:
            # Merge with original data to get moving average
            merged_results = greykite_results.merge(
                df_group[['WEEK_BEGIN', 'WEEKLY_ATTENDANCE_RATE', 'FORECAST_ATTENDANCE_RATE_WITH_OUTLIER', 'WEEK_NUMBER']], 
                on='WEEK_BEGIN', 
                how='left'
            )
            
            # Add group identifiers
            merged_results['WORK_LOCATION'] = work_location
            merged_results['SHIFT_TIME'] = shift_time
            merged_results['DEPARTMENT_GROUP'] = department_group
            
            # Rename columns for clarity
            merged_results = merged_results.rename(columns={
                'WEEKLY_ATTENDANCE_RATE': 'ACTUAL_ATTENDANCE_RATE_ORIG',
                'FORECAST_ATTENDANCE_RATE_WITH_OUTLIER': 'MOVING_AVG_4WEEK_FORECAST'
            })
            
            # Ensure we have the actual values (they should match between ACTUAL_ATTENDANCE_RATE and ACTUAL_ATTENDANCE_RATE_ORIG)
            merged_results['ACTUAL_ATTENDANCE_RATE'] = merged_results['ACTUAL_ATTENDANCE_RATE'].fillna(
                merged_results['ACTUAL_ATTENDANCE_RATE_ORIG']
            )
            
            all_results.append(merged_results)
            
            # Calculate and print accuracy metrics
            valid_data = merged_results.dropna(subset=['ACTUAL_ATTENDANCE_RATE', 'GREYKITE_FORECAST', 'MOVING_AVG_4WEEK_FORECAST'])
            
            if len(valid_data) > 0:
                mae_greykite = mean_absolute_error(valid_data['ACTUAL_ATTENDANCE_RATE'], valid_data['GREYKITE_FORECAST'])
                mae_moving_avg = mean_absolute_error(valid_data['ACTUAL_ATTENDANCE_RATE'], valid_data['MOVING_AVG_4WEEK_FORECAST'])
                
                print(f"  Greykite MAE: {mae_greykite:.4f}")
                print(f"  Moving Avg MAE: {mae_moving_avg:.4f}")
        else:
            print(f"  Failed to generate Greykite forecast")
    
    # Combine all results
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        
        # Select and order columns for output
        output_columns = [
            'WEEK_BEGIN', 'WEEK_NUMBER', 'WORK_LOCATION', 'SHIFT_TIME', 'DEPARTMENT_GROUP',
            'ACTUAL_ATTENDANCE_RATE', 'GREYKITE_FORECAST', 'MOVING_AVG_4WEEK_FORECAST'
        ]
        
        # Only include columns that exist
        available_output_columns = [col for col in output_columns if col in final_results.columns]
        final_results = final_results[available_output_columns]
        
        # Sort by date and grouping columns
        sort_columns = ['WEEK_BEGIN', 'WORK_LOCATION', 'SHIFT_TIME']
        if 'DEPARTMENT_GROUP' in final_results.columns:
            sort_columns.append('DEPARTMENT_GROUP')
        
        final_results = final_results.sort_values(sort_columns)
        
        # Save to CSV
        output_filename = f"attendance_forecast_results_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        final_results.to_csv(output_filename, index=False)
        
        print(f"\n=== RESULTS SUMMARY ===")
        print(f"Total records processed: {len(final_results)}")
        print(f"Unique combinations: {len(final_results.groupby(['WORK_LOCATION', 'SHIFT_TIME', 'DEPARTMENT_GROUP']))}")
        print(f"Date range: {final_results['WEEK_BEGIN'].min()} to {final_results['WEEK_BEGIN'].max()}")
        print(f"Results saved to: {output_filename}")
        
        # Display sample results
        print(f"\nSample results:")
        print(final_results.head(10).to_string(index=False))
        
        # Overall accuracy summary
        valid_final = final_results.dropna(subset=['ACTUAL_ATTENDANCE_RATE', 'GREYKITE_FORECAST', 'MOVING_AVG_4WEEK_FORECAST'])
        if len(valid_final) > 0:
            overall_mae_greykite = mean_absolute_error(valid_final['ACTUAL_ATTENDANCE_RATE'], valid_final['GREYKITE_FORECAST'])
            overall_mae_moving_avg = mean_absolute_error(valid_final['ACTUAL_ATTENDANCE_RATE'], valid_final['MOVING_AVG_4WEEK_FORECAST'])
            
            print(f"\n=== OVERALL ACCURACY ===")
            print(f"Overall Greykite MAE: {overall_mae_greykite:.4f}")
            print(f"Overall Moving Average MAE: {overall_mae_moving_avg:.4f}")
            
            if overall_mae_greykite < overall_mae_moving_avg:
                print("Greykite performs better overall")
            else:
                print("Moving Average performs better overall")
    
    else:
        print("No results generated. Please check your data and requirements.")

if __name__ == "__main__":
    main() 