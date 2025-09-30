#!/usr/bin/env python3
"""
Enhanced Attendance Forecasting Script using Greykite
- Incorporates HelloFresh-style implementation patterns
- Uses US holiday effects and prediction intervals
- Comprehensive evaluation with cross-validation
- Thursday week start alignment
- Rolling 1-week forecast horizon with gap rule
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
import warnings
from collections import defaultdict
import plotly.io
from sklearn.metrics import mean_absolute_error

# Greykite imports
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.templates.autogen.forecast_config import ForecastConfig, MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam, EvaluationMetricParam, EvaluationPeriodParam
from greykite.framework.utils.result_summary import summarize_grid_search_results

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', None)

def week_to_thursday_date(week_str: str) -> pd.Timestamp:
    """Convert ISO-week string 'YYYY-Www' to the Thursday date of that week."""
    year, week = map(int, week_str.split('-W'))
    # ISO week starts on Monday, so Thursday is day 4
    thursday = date.fromisocalendar(year, week, 4)
    return pd.Timestamp(thursday)

def prepare_data_for_greykite(df_group, actual_col='WEEKLY_ATTENDANCE_RATE'):
    """
    Prepare data for Greykite with Thursday week alignment and proper formatting.
    """
    df = df_group.dropna(subset=[actual_col]).copy()
    
    # Convert week numbers to Thursday dates (HelloFresh style)
    df['WEEK_BEGIN'] = df['WEEK_NUMBER'].apply(week_to_thursday_date)
    df['y'] = df[actual_col]
    
    # Sort by date and reset index
    df = df.sort_values('WEEK_BEGIN').reset_index(drop=True)
    
    # Keep only necessary columns for Greykite
    greykite_df = df[['WEEK_BEGIN', 'y', 'WEEK_NUMBER']].copy()
    
    return greykite_df, df

def run_enhanced_greykite_forecast(df_group, actual_col='WEEKLY_ATTENDANCE_RATE', 
                                 forecast_horizon=5, coverage=0.95, plot=False,
                                 target_year='2025', min_week=None):
    """
    Enhanced Greykite forecasting with HelloFresh-style implementation.
    
    Parameters:
    - df_group: DataFrame with attendance data
    - actual_col: Column name for actual attendance values
    - forecast_horizon: Number of weeks to forecast ahead
    - coverage: Prediction interval coverage (0.95 = 95%)
    - plot: Whether to generate plots
    - target_year: Year to focus forecasts on
    
    Returns:
    - Dictionary with forecast results, metrics, and diagnostics
    """
    
    # Prepare data
    greykite_df, original_df = prepare_data_for_greykite(df_group, actual_col)
    
    if len(greykite_df) < 20:  # Need sufficient data for cross-validation
        print(f"  Insufficient data: {len(greykite_df)} weeks (minimum 20 required)")
        return None
    
    print(f"  === ENHANCED GREYKITE ANALYSIS ===")
    print(f"  Total historical data: {len(greykite_df)} weeks")
    print(f"  Date range: {greykite_df['WEEK_BEGIN'].min().strftime('%Y-%m-%d')} to {greykite_df['WEEK_BEGIN'].max().strftime('%Y-%m-%d')}")
    print(f"  Attendance stats: Mean={greykite_df['y'].mean():.2f}%, Std={greykite_df['y'].std():.2f}%")
    print(f"  Forecast horizon: {forecast_horizon} weeks")
    print(f"  Prediction interval coverage: {coverage*100:.0f}%")
    
    # Configure metadata with Thursday week start (HelloFresh style)
    metadata = MetadataParam(
        time_col="WEEK_BEGIN",
        value_col="y",
        freq="W-THU"  # Thursday week start
    )
    
    # Enhanced model components with US holiday effects
    model_components = ModelComponentsParam(
        events={
            "holiday_lookup_countries": ["US"],
            "holiday_pre_num_days": 8,
            "holiday_post_num_days": 2,
        },
        uncertainty={
            "uncertainty_dict": {
                "uncertainty_method": "quantile",
                "params": {
                    "coverage": [0.80, 0.95],
                    "calibrate": True,   # <-- key
                    "n_boot": 200
                }
            }
        }
    )

    evaluation_period = EvaluationPeriodParam(
        cv_max_splits=8,                 # rolling origins
        cv_min_train_periods=52,         # ≥ 1y
        cv_expanding_window=True,        # expanding window
        periods_between_train_test=1,    # gap = 1 week blackout
        cv_horizon=2,                    # evaluate T+2
        test_horizon=2,
        cv_use_most_recent_splits=True
    )

    # Enhanced evaluation configuration
    evaluation_config = EvaluationMetricParam(
        cv_selection_metric="MeanAbsoluteError",  # Use standard metric name
        cv_report_metrics=[
            "MeanAbsoluteError",
            "RootMeanSquaredError",
            "Correlation"
        ]
    )
    
    # Create forecast configuration
    config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=forecast_horizon,
        coverage=coverage,  # Prediction intervals
        metadata_param=metadata,
        model_components_param=model_components,
        evaluation_metric_param=evaluation_config,
        evaluation_period_param=evaluation_period
    )
    
    # Initialize forecaster and run
    forecaster = Forecaster()
    
    print(f"  Running Greykite with cross-validation...")
    try:
        result = forecaster.run_forecast_config(
            df=greykite_df,
            config=config
        )
        
        print(f"  ✓ Greykite model completed successfully")
        
        # Extract results
        ts = result.timeseries
        backtest = result.backtest
        grid_search = result.grid_search
        forecast = result.forecast
        model = result.model
        
        # Comprehensive evaluation metrics
        print(f"\n  === CROSS-VALIDATION RESULTS ===")
        cv_results = summarize_grid_search_results(
            grid_search=grid_search,
            decimals=3,
            cv_report_metrics=["MeanAbsoluteError", "RootMeanSquaredError"],
            column_order=["rank", "mean_test", "mean_train", "mean_fit_time", "params"]
        )
        
        if not cv_results.empty:
            print(f"  Best model CV metrics:")
            best_row = cv_results.iloc[0]
            print(f"    MAE: {best_row.get('mean_test_MeanAbsoluteError', 'N/A')}")
            print(f"    RMSE: {best_row.get('mean_test_RootMeanSquaredError', 'N/A')}")
        
        # Backtest evaluation
        print(f"\n  === BACKTEST EVALUATION ===")
        backtest_metrics = defaultdict(list)
        if hasattr(backtest, 'train_evaluation') and hasattr(backtest, 'test_evaluation'):
            for metric, value in backtest.train_evaluation.items():
                backtest_metrics[metric].append(value)
                backtest_metrics[metric].append(backtest.test_evaluation[metric])
            
            metrics_df = pd.DataFrame(backtest_metrics, index=["train", "test"]).T
            print(f"  Train vs Test metrics:")
            for metric in ['MeanAbsoluteError', 'RootMeanSquaredError']:
                if metric in metrics_df.index:
                    train_val = metrics_df.loc[metric, 'train']
                    test_val = metrics_df.loc[metric, 'test']
                    print(f"    {metric}: Train={train_val:.3f}, Test={test_val:.3f}")
        
        # Generate rolling 1-week forecasts for target year
        print(f"\n  === GENERATING ROLLING T+2 FORECASTS WITH GAP=1 ===")

        target_weeks = original_df[original_df['WEEK_NUMBER'].str.startswith(target_year)].copy()
        # If a minimum week threshold is provided (e.g., '2025-W28'), filter to that and later
        if isinstance(min_week, str) and min_week[:4] == str(target_year):
            target_weeks = target_weeks[target_weeks['WEEK_NUMBER'] >= min_week]
        if target_weeks.empty:
            print(f"  No {target_year} weeks found in data")
            future_results = pd.DataFrame()
        else:
            print(f"  Generating T+2 forecasts for {len(target_weeks)} weeks in {target_year}")
            rolling_forecasts = []
            forecaster_local = Forecaster()  # new forecaster per segment loop is fine

            MIN_TRAIN_WEEKS = 52
            for i, (_, target_row) in enumerate(target_weeks.iterrows()):
                T = target_row['WEEK_BEGIN']              # target week ts
                cutoff = T - pd.Timedelta(weeks=2)        # train up to T-2 (gap=1 → blackout week T-1)

                # strict train slice
                train_df = greykite_df[greykite_df['WEEK_BEGIN'] <= cutoff].copy()
                if len(train_df) < MIN_TRAIN_WEEKS:
                    continue

                # model config: horizon=2 so we can take the 2nd step (index 1)
                local_config = ForecastConfig(
                    model_template=ModelTemplateEnum.SILVERKITE.name,
                    forecast_horizon=2,
                    coverage=coverage,
                    metadata_param=metadata,
                    model_components_param=model_components,
                    evaluation_metric_param=evaluation_config
                )

                try:
                    local_result = forecaster_local.run_forecast_config(df=train_df, config=local_config)
                    fdf = local_result.forecast.df
                    if len(fdf) >= 2:
                        # index 1 is T+2 relative to cutoff
                        row2 = fdf.iloc[1]
                        rolling_forecasts.append({
                            'WEEK_BEGIN': T,
                            'WEEK_NUMBER': target_row['WEEK_NUMBER'],
                            'forecast': row2.get('forecast', np.nan),
                            'forecast_lower': row2.get('forecast_lower', np.nan) if 'forecast_lower' in fdf.columns else np.nan,
                            'forecast_upper': row2.get('forecast_upper', np.nan) if 'forecast_upper' in fdf.columns else np.nan
                        })
                except Exception as e:
                    print(f"    Error forecasting {target_row['WEEK_NUMBER']}: {str(e)[:120]}")

                if (i + 1) % 10 == 0:
                    print(f"    Completed {i + 1}/{len(target_weeks)} targets")

            future_results = pd.DataFrame(rolling_forecasts)
        
        if len(future_results) > 0:
            print(f"  Generated {len(future_results)} rolling forecasts for {target_year}")
            print(f"  Forecast range: {future_results['forecast'].min():.2f}% to {future_results['forecast'].max():.2f}%")
            print(f"  Forecast mean: {future_results['forecast'].mean():.2f}%")
            
            if 'forecast_lower' in future_results.columns and future_results['forecast_lower'].notna().any():
                valid_intervals = future_results.dropna(subset=['forecast_lower', 'forecast_upper'])
                if len(valid_intervals) > 0:
                    avg_interval_width = (valid_intervals['forecast_upper'] - valid_intervals['forecast_lower']).mean()
                    print(f"  Average prediction interval width: {avg_interval_width:.2f}%")
        else:
            print(f"  No forecasts generated for {target_year}")
        
        # Prepare return results
        results = {
            'future_forecasts': future_results,
            'cv_results': cv_results,
            'backtest_metrics': metrics_df if 'metrics_df' in locals() else None,
            'model_result': result,
            'data_summary': {
                'total_weeks': len(greykite_df),
                'date_range': (greykite_df['WEEK_BEGIN'].min(), greykite_df['WEEK_BEGIN'].max()),
                'attendance_stats': {
                    'mean': greykite_df['y'].mean(),
                    'std': greykite_df['y'].std(),
                    'min': greykite_df['y'].min(),
                    'max': greykite_df['y'].max()
                }
            }
        }
        
        return results
        
    except Exception as e:
        print(f"  ✗ Error in Greykite forecasting: {str(e)}")
        return None

def compare_with_simple_baselines(df_group, target_year='2025', actual_col='WEEKLY_ATTENDANCE_RATE'):
    """
    Compare Greykite results with simple baseline methods.
    """
    print(f"\n  === BASELINE COMPARISON ===")
    
    df = df_group.dropna(subset=[actual_col]).copy()
    df['WEEK_BEGIN'] = df['WEEK_NUMBER'].apply(week_to_thursday_date)
    df['y'] = df[actual_col]
    df = df.sort_values('WEEK_BEGIN').reset_index(drop=True)
    
    # Find target year data for comparison
    target_weeks = df[df['WEEK_NUMBER'].str.startswith(target_year)].copy()
    if target_weeks.empty:
        print(f"  No data found for {target_year}")
        return None
    
    baselines = {}
    
    # Historical mean baseline
    historical_mean = df['y'].mean()
    baselines['historical_mean'] = {
        'forecast': [historical_mean] * len(target_weeks),
        'mae': mean_absolute_error(target_weeks['y'], [historical_mean] * len(target_weeks))
    }
    
    # Recent 12-week mean
    recent_mean = df['y'].tail(12).mean()
    baselines['recent_12w_mean'] = {
        'forecast': [recent_mean] * len(target_weeks),
        'mae': mean_absolute_error(target_weeks['y'], [recent_mean] * len(target_weeks))
    }
    
    # Recent 8-week median
    recent_median = df['y'].tail(8).median()
    baselines['recent_8w_median'] = {
        'forecast': [recent_median] * len(target_weeks),
        'mae': mean_absolute_error(target_weeks['y'], [recent_median] * len(target_weeks))
    }
    
    print(f"  Baseline MAE comparison:")
    for name, metrics in baselines.items():
        print(f"    {name}: {metrics['mae']:.2f}%")
    
    return baselines

def main():
    """
    Main execution function with enhanced Greykite implementation.
    """
    print("=== ENHANCED ATTENDANCE FORECASTING WITH GREYKITE ===")
    print("Incorporating HelloFresh-style patterns:")
    print("- US holiday effects")
    print("- Prediction intervals") 
    print("- Cross-validation")
    print("- Thursday week alignment")
    print("- Rolling 1-week forecasting with gap rule")
    print("- Processing ALL combinations of location, shift, and department")
    print("- Generating forecasts for BOTH 2024 and 2025")
    
    # Load data
    input_file = '/Users/nikhil.ranka/attendance-analytics-dashboard/Labor_Management-Greykite_Input.csv'
    
    try:
        df = pd.read_csv(input_file)
        print(f"\n✓ Loaded data: {len(df)} records from {input_file}")
        print(f"Available columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"✗ Error: Could not find {input_file}")
        return
    
    # Identify actuals column (prefer new name with legacy fallbacks)
    preferred_actual_cols = [
        'WEEKLY_ATTENDANCE_RATE',
        'Weekly_attendance_rate',
        'ATTENDANCE_RATE_WITH_OUTLIER'
    ]
    actual_col = None
    for col in preferred_actual_cols:
        if col in df.columns:
            actual_col = col
            break
    if actual_col is None:
        print("✗ Error: Could not find an attendance rate column. Expected one of:", preferred_actual_cols)
        return

    print(f"Using actuals column: '{actual_col}'")

    # Identify all unique combinations
    grouping_cols = ['WORK_LOCATION', 'SHIFT_TIME', 'DEPARTMENT_GROUP']
    available_cols = [col for col in grouping_cols if col in df.columns]
    
    if not available_cols:
        print("✗ Error: Required grouping columns not found in data")
        return
    
    print(f"\nGrouping by: {available_cols}")
    
    # Get all unique combinations
    combinations = df[available_cols].drop_duplicates().reset_index(drop=True)
    print(f"Found {len(combinations)} unique combinations to process")
    
    # Display combinations
    for i, (_, combo) in enumerate(combinations.iterrows()):
        combo_str = ", ".join([f"{col}='{combo[col]}'" for col in available_cols])
        print(f"  {i+1}. {combo_str}")
    
    # Restrict to NJ Newark Lister Ave, PM, Production only
    combinations = combinations[
        (combinations['WORK_LOCATION'] == 'NJ Newark Lister Ave') &
        (combinations['SHIFT_TIME'] == 'PM') &
        (combinations['DEPARTMENT_GROUP'] == 'Production')
    ].reset_index(drop=True)

    # Process filtered combinations
    all_output_data = []
    successful_combinations = 0
    
    for combo_idx, (_, combination) in enumerate(combinations.iterrows()):
        print(f"\n" + "="*80)
        print(f"PROCESSING COMBINATION {combo_idx + 1}/{len(combinations)}")
        combo_str = ", ".join([f"{col}='{combination[col]}'" for col in available_cols])
        print(f"{combo_str}")
        print(f"="*80)
        
        # Filter data for this combination
        filtered_df = df.copy()
        for col in available_cols:
            filtered_df = filtered_df[filtered_df[col] == combination[col]]
        
        print(f"Filtered data: {len(filtered_df)} records")
        
        if len(filtered_df) < 20:
            print("✗ Insufficient data for analysis (minimum 20 records required)")
            continue
        
        # Run enhanced Greykite forecasting for both 2024 and 2025 (we will filter inside to 2025-W28+)
        target_years = ['2024', '2025']
        all_results = {}
        combination_successful = False
        
        for target_year in target_years:
            print(f"    Processing {target_year} forecasts...")
            min_week_threshold = '2025-W28' if target_year == '2025' else None
            results = run_enhanced_greykite_forecast(
                df_group=filtered_df,
                actual_col=actual_col,
                forecast_horizon=1,  # 1-week horizon for weekly planning
                coverage=0.95,  # 95% prediction intervals
                target_year=target_year,
                min_week=min_week_threshold
            )
            
            if results is not None:
                all_results[target_year] = results
                combination_successful = True
                print(f"    ✓ {target_year} forecasts completed")
            else:
                print(f"    ✗ {target_year} forecasting failed")
        
        if not combination_successful:
            print("✗ Greykite forecasting failed for all years in this combination")
            continue
        
        successful_combinations += 1
        
        # Compare with baselines for both years
        all_baselines = {}
        for target_year in target_years:
            if target_year in all_results:
                baselines = compare_with_simple_baselines(filtered_df, target_year=target_year, actual_col=actual_col)
                if baselines:
                    all_baselines[target_year] = baselines
        
        # Prepare output data for this combination (process all years)
        for target_year, results in all_results.items():
            if results and 'future_forecasts' in results:
                future_forecasts = results['future_forecasts']
                
                for _, row in future_forecasts.iterrows():
                    # Find matching row in original data to get actual values and other forecasts
                    original_row = filtered_df[filtered_df['WEEK_NUMBER'] == row['WEEK_NUMBER']]
                    
                    if not original_row.empty:
                        orig = original_row.iloc[0]
                        
                        # Create output row with actual combination values
                        output_row = {
                            'week_number': row['WEEK_NUMBER']
                        }
                        
                        # Add combination identifiers
                        for col in available_cols:
                            output_row[col.lower()] = combination[col]
                        
                        # Add forecasting results
                        output_row.update({
                            'actual_attendance': orig.get(actual_col, None),
                            'greykite_forecast': row['forecast']
                        })
                        
                        # Add prediction intervals if available
                        if 'forecast_lower' in row:
                            output_row['greykite_forecast_lower_95'] = row['forecast_lower']
                            output_row['greykite_forecast_upper_95'] = row['forecast_upper']
                        
                        # Add other forecasting models from input data
                        other_forecast_columns = [
                            'FOUR_WEEK_ROLLING_AVG',
                            'SIX_WEEK_ROLLING_AVG', 
                            'EXPONENTIAL_SMOOTHING_0_2',
                            'EXPONENTIAL_SMOOTHING_0_4',
                            'EXPONENTIAL_SMOOTHING_0_6',
                            'EXPONENTIAL_SMOOTHING_0_8',
                            'EXPONENTIAL_SMOOTHING_1'
                        ]
                        
                        for col in other_forecast_columns:
                            if col in orig:
                                # Use more readable column names
                                clean_name = col.lower().replace('_', '_')
                                if 'exponential' in clean_name:
                                    # Extract alpha value for cleaner naming
                                    alpha = col.split('_')[-1].replace('_', '.')
                                    clean_name = f'exponential_smoothing_alpha_{alpha}'
                                output_row[clean_name] = orig[col]
                        
                        all_output_data.append(output_row)
                    else:
                        # If no original data found, still include Greykite forecast
                        output_row = {
                            'week_number': row['WEEK_NUMBER']
                        }
                        
                        # Add combination identifiers
                        for col in available_cols:
                            output_row[col.lower()] = combination[col]
                        
                        # Add forecasting results
                        output_row.update({
                            'actual_attendance': None,
                            'greykite_forecast': row['forecast']
                        })
                        
                        if 'forecast_lower' in row:
                            output_row['greykite_forecast_lower_95'] = row['forecast_lower']
                            output_row['greykite_forecast_upper_95'] = row['forecast_upper']
                        
                        all_output_data.append(output_row)
    
    print(f"\n" + "="*80)
    print(f"PROCESSING COMPLETE")
    print(f"Successfully processed: {successful_combinations}/{len(combinations)} combinations")
    print(f"="*80)
    
    # Create output DataFrame
    if all_output_data:
        output_df = pd.DataFrame(all_output_data)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"enhanced_attendance_forecast_all_combinations_2024_2025_{timestamp}.csv"
        
        # Save results
        output_df.to_csv(output_file, index=False)
        print(f"\n✓ Enhanced forecasts saved to: {output_file}")
        print(f"  Records: {len(output_df)}")
        print(f"  Combinations processed: {successful_combinations}")
        print(f"  Columns: {list(output_df.columns)}")
        
        # Display sample results
        print(f"\nSample forecast results (first 10 records):")
        # Show key columns for readability
        display_cols = ['week_number']
        
        # Add grouping columns
        for col in available_cols:
            col_name = col.lower()
            if col_name in output_df.columns:
                display_cols.append(col_name)
        
        # Add forecast columns
        forecast_cols = ['actual_attendance', 'greykite_forecast']
        if 'greykite_forecast_lower_95' in output_df.columns:
            forecast_cols.extend(['greykite_forecast_lower_95', 'greykite_forecast_upper_95'])
        
        display_cols.extend(forecast_cols)
        
        # Add a couple other forecast models for comparison
        other_models = ['four_week_rolling_avg', 'exponential_smoothing_alpha_0.4']
        for col in other_models:
            if col in output_df.columns:
                display_cols.append(col)
        
        # Filter display columns to only those that exist
        display_cols = [col for col in display_cols if col in output_df.columns]
        
        print(output_df[display_cols].head(10).to_string(index=False, float_format='%.2f'))
        
        # Summary statistics
        print(f"\nOverall Forecast Summary:")
        print(f"  Total forecasts generated: {len(output_df)}")
        print(f"  Greykite forecast: Mean={output_df['greykite_forecast'].mean():.2f}%, Std={output_df['greykite_forecast'].std():.2f}%")
        print(f"  Greykite range: {output_df['greykite_forecast'].min():.2f}% to {output_df['greykite_forecast'].max():.2f}%")
        
        if 'actual_attendance' in output_df.columns and output_df['actual_attendance'].notna().any():
            actual_data = output_df.dropna(subset=['actual_attendance'])
            print(f"  Actual attendance: Mean={actual_data['actual_attendance'].mean():.2f}%, Std={actual_data['actual_attendance'].std():.2f}%")
            
            # Calculate MAE for Greykite vs actual
            greykite_mae = abs(actual_data['greykite_forecast'] - actual_data['actual_attendance']).mean()
            print(f"  Overall Greykite MAE vs Actual: {greykite_mae:.2f}%")
        
        if 'greykite_forecast_lower_95' in output_df.columns:
            valid_intervals = output_df.dropna(subset=['greykite_forecast_lower_95', 'greykite_forecast_upper_95'])
            if len(valid_intervals) > 0:
                avg_interval = (valid_intervals['greykite_forecast_upper_95'] - valid_intervals['greykite_forecast_lower_95']).mean()
                print(f"  Average 95% prediction interval width: {avg_interval:.2f}%")
        
        # Summary by combination
        print(f"\nSummary by Combination:")
        grouping_cols_lower = [col.lower() for col in available_cols if col.lower() in output_df.columns]
        if grouping_cols_lower:
            combo_summary = output_df.groupby(grouping_cols_lower).agg({
                'greykite_forecast': ['count', 'mean', 'std'],
                'actual_attendance': 'mean'
            }).round(2)
            print(combo_summary.head(10))
        
        # Compare forecast models if available
        forecast_models = [col for col in output_df.columns if 'forecast' in col.lower() or 'smoothing' in col.lower() or 'rolling' in col.lower()]
        forecast_models = [col for col in forecast_models if col not in ['greykite_forecast', 'greykite_forecast_lower_95', 'greykite_forecast_upper_95']]
        
        if forecast_models and 'actual_attendance' in output_df.columns:
            print(f"\nModel Comparison (MAE vs Actual):")
            actual_data = output_df.dropna(subset=['actual_attendance'])
            if len(actual_data) > 0:
                for model in forecast_models[:5]:  # Show top 5 models
                    if model in actual_data.columns and actual_data[model].notna().any():
                        model_mae = abs(actual_data[model] - actual_data['actual_attendance']).mean()
                        print(f"  {model}: {model_mae:.2f}%")
    else:
        print("\n✗ No forecasts generated - no combinations had sufficient data")
    
    print(f"\n" + "="*80)
    print(f"ENHANCED GREYKITE ANALYSIS COMPLETE")
    print(f"Processed {successful_combinations} combinations successfully")
    print(f"="*80)

if __name__ == "__main__":
    main()