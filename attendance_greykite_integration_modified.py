#!/usr/bin/env python3
"""
Modified Attendance Greykite Integration Script
- Keeps original OUTPUT format and column names
- Retains improved logic (T+2 horizon with 1-week gap, CV alignment, AR lags, post-holiday, winsorization, leakage-safe baseline)
- Adds robust week/date handling and baseline fallback
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import warnings
from sklearn.metrics import mean_absolute_error

# Greykite imports
from greykite.framework.templates.autogen.forecast_config import (
    EvaluationPeriodParam, ForecastConfig, MetadataParam, ModelComponentsParam,
    EvaluationMetricParam
)
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum

warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = (12, 6)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', None)

# ---------------------------
# HelloFresh week helpers (unchanged signatures)
# ---------------------------
def datetime_to_hf_week(date_time: dt.datetime):
    hf_datetime = date_time + dt.timedelta(days=2)
    year, week, _ = hf_datetime.isocalendar()
    return f'{year}-W{week:02d}'

def hf_week_to_datetime(hf_week: str):
    return dt.datetime.strptime(hf_week + '-1', '%G-W%V-%w') - dt.timedelta(days=2)

def hf_week_add(hf_week: str, num_weeks: int):
    return datetime_to_hf_week(hf_week_to_datetime(hf_week) + dt.timedelta(weeks=num_weeks))

def hf_week_sub(hf_week: str, num_weeks: int):
    return datetime_to_hf_week(hf_week_to_datetime(hf_week) - dt.timedelta(weeks=num_weeks))

def current_hf_week():
    return datetime_to_hf_week(dt.datetime.today())

def hf_week_to_last_thursday_str(hf_week: str):
    monday = hf_week_to_datetime(hf_week) - dt.timedelta(days=2)
    return monday.strftime('%Y-%m-%d')

# ---------------------------
# New: robust week/date utilities (internal only; output unchanged)
# ---------------------------
def safe_week_str(ts: pd.Timestamp, fmt: str = "%Y-W%V"):
    try:
        return ts.strftime(fmt) if pd.notna(ts) else "NaT"
    except Exception:
        return "NaT"

def coerce_week_begin(df: pd.DataFrame,
                      week_col: str = "WEEK_BEGIN",
                      weeknum_col: str = "WEEK_NUMBER") -> pd.DataFrame:
    """
    - Parse WEEK_BEGIN if present
    - If missing or NaT but WEEK_NUMBER exists, rebuild from HF week string
    - Keep the column name WEEK_BEGIN (for internal use only)
    """
    if week_col in df.columns:
        df[week_col] = pd.to_datetime(df[week_col], errors="coerce")
    else:
        df[week_col] = pd.NaT

    if weeknum_col in df.columns:
        mask = df[week_col].isna() & df[weeknum_col].notna()
        if mask.any():
            df.loc[mask, week_col] = df.loc[mask, weeknum_col].apply(hf_week_to_datetime)

    return df

# ---------------------------
# Baseline calc (keeps SAME column name)
# ---------------------------
def calculate_moving_average(df, window=4, attendance_col='WEEKLY_ATTENDANCE_RATE', shift=0):
    """
    Rolling mean with optional shift to avoid leakage.
    For weekly T+2 with 1-week blackout, use shift=2.
    NOTE: We will still write the output column as MOVING_AVG_4WEEK_FORECAST to preserve format.
    """
    return df[attendance_col].rolling(window=window, min_periods=1).mean().shift(shift)

# ---------------------------
# Baseline fallback (produces same output cols as before)
# ---------------------------
def baseline_realistic_forecast(df_group, actual_col, forecast_start_date='2024-01-01', gap_weeks=1, window=4):
    df_group = df_group.sort_values('WEEK_BEGIN').reset_index(drop=True)
    forecast_start = pd.to_datetime(forecast_start_date)

    shift_steps = gap_weeks + 1  # =2 for weekly T+2
    df_group["__BASELINE_MA4_S2"] = (
        df_group[actual_col].rolling(window, min_periods=1).mean().shift(shift_steps)
    )

    target_rows = df_group[df_group["WEEK_BEGIN"] >= forecast_start].copy()
    if target_rows.empty:
        return None

    rows = []
    for _, r in target_rows.iterrows():
        target_week = r["WEEK_BEGIN"]
        cutoff = target_week - pd.Timedelta(weeks=gap_weeks + 1)
        train_slice = df_group[df_group["WEEK_BEGIN"] <= cutoff]
        if train_slice.empty:
            continue

        rows.append({
            "WEEK_BEGIN": target_week,
            "ACTUAL_ATTENDANCE_RATE": r[actual_col],
            "GREYKITE_FORECAST": np.nan,  # fallback has no GK
            "MOVING_AVG_4WEEK_FORECAST": r["__BASELINE_MA4_S2"],  # same name as before
            "TRAINING_WEEKS_USED": len(train_slice),
            "TRAINING_END_DATE": train_slice["WEEK_BEGIN"].max(),
            "GAP_WEEKS": gap_weeks
        })

    out = pd.DataFrame(rows) if rows else None
    return out

# ---------------------------
# Greykite realistic forecast (improved logic; output columns unchanged)
# ---------------------------
def run_greykite_realistic_forecast(df_group, actual_col='WEEKLY_ATTENDANCE_RATE',
                                    forecast_start_date='2024-01-01', gap_weeks=1):
    MIN_TRAIN_WEEKS = 52  # keep strong minimum; fallback covers thin segments

    df_group = df_group.dropna(subset=['WEEK_BEGIN']).sort_values('WEEK_BEGIN').reset_index(drop=True)
    if len(df_group) < MIN_TRAIN_WEEKS:
        print(f"    Skipping Greykite: insufficient total history ({len(df_group)} < {MIN_TRAIN_WEEKS})")
        return None

    forecast_start = pd.to_datetime(forecast_start_date)

    # Robustness: winsorize target per segment (1–99 pct)
    lo, hi = df_group[actual_col].quantile([0.01, 0.99])
    df_group[actual_col] = df_group[actual_col].clip(lo, hi)

    # Leakage-safe shifted rolling features used as regressors (shift=2)
    shift_steps = gap_weeks + 1
    df_group["ROLL_MEAN_4_S2"] = df_group[actual_col].rolling(4, min_periods=1).mean().shift(shift_steps)
    df_group["ROLL_MEDIAN_4_S2"] = df_group[actual_col].rolling(4, min_periods=1).median().shift(shift_steps)

    forecast_weeks = df_group[df_group['WEEK_BEGIN'] >= forecast_start].copy()
    if len(forecast_weeks) == 0:
        print(f"    No data found from {forecast_start_date} onwards")
        return None

    print(f"    Running realistic Greykite for {len(forecast_weeks)} target weeks (gap={gap_weeks})")

    results = []
    forecaster = Forecaster()

    for _, row in forecast_weeks.iterrows():
        target_week = row['WEEK_BEGIN']
        actual_value = row[actual_col]

        training_cutoff = target_week - pd.Timedelta(weeks=gap_weeks + 1)
        train_data = df_group[df_group['WEEK_BEGIN'] <= training_cutoff].copy()
        train_data = train_data.dropna(subset=['WEEK_BEGIN', actual_col])

        if len(train_data) < MIN_TRAIN_WEEKS:
            print(f"    Skip target {safe_week_str(target_week)}: train weeks {len(train_data)} < {MIN_TRAIN_WEEKS}")
            continue

        print(
            f"    Forecasting {safe_week_str(target_week)} using {len(train_data)} weeks up to "
            f"{safe_week_str(train_data['WEEK_BEGIN'].max())} (gap={gap_weeks})"
        )

        metadata = MetadataParam(
            time_col="WEEK_BEGIN",
            value_col=actual_col,
            freq="W-THU"  # keep original anchor to avoid any output/behavior change you don't want
        )

        model_components = ModelComponentsParam(
            seasonality={
                "yearly_seasonality": {
                    "seas_names": ["yearly"],
                    "fourier_series": {"yearly": {"period": 52.18, "order": 4}}
                }
                # weekly_seasonality: removed (not identifiable at weekly sampling)
            },
            autoregression={"autoreg_dict": {"autoreg_orders": [1, 2, 3, 4, 52]}},
            events={
                "holiday_lookup_countries": ["US"],
                "holiday_pre_num_days": 8,
                "holiday_post_num_days": 3
            },
            regressors={"regressor_cols": ["ROLL_MEAN_4_S2", "ROLL_MEDIAN_4_S2"]}
        )

        evaluation = EvaluationPeriodParam(
            test_horizon=2,                         # exactly T+2
            periods_between_train_test=1,           # 1-week blackout
            cv_horizon=2,
            cv_min_train_periods=MIN_TRAIN_WEEKS,
            cv_expanding_window=True,
            cv_use_most_recent_splits=True,
            cv_periods_between_splits=1,
            cv_periods_between_train_test=1,
            cv_max_splits=3
        )

        try:
            result = forecaster.run_forecast_config(
                df=train_data,
                config=ForecastConfig(
                    model_template=ModelTemplateEnum.AUTO.name,
                    forecast_horizon=gap_weeks + 1,  # =2; take step index 1
                    coverage=0.95,
                    metadata_param=metadata,
                    model_components_param=model_components,
                    evaluation_metric_param=EvaluationMetricParam(
                        cv_selection_metric="MeanAbsoluteError"
                    ),
                    evaluation_period_param=evaluation
                )
            )

            forecast_df = result.forecast.df
            if len(forecast_df) > gap_weeks:
                forecast_value = forecast_df.iloc[gap_weeks]['forecast']
                results.append({
                    'WEEK_BEGIN': target_week,
                    'ACTUAL_ATTENDANCE_RATE': actual_value,
                    'GREYKITE_FORECAST': forecast_value,
                    'TRAINING_WEEKS_USED': len(train_data),
                    'TRAINING_END_DATE': train_data['WEEK_BEGIN'].max(),
                    'GAP_WEEKS': gap_weeks
                })

        except Exception as e:
            print(f"      Greykite error at {safe_week_str(target_week)}: {e}")
            continue

    return pd.DataFrame(results) if results else None

# ---------------------------
# Main
# ---------------------------
def main():
    """Main function to run the modified attendance forecasting script"""

    csv_path = "/Users/nikhil.ranka/attendance-analytics-dashboard/Labor_Management-Greykite_Input.csv"

    try:
        df_attendance = pd.read_csv(csv_path)
        print(f"Successfully loaded data from {csv_path}")
        print(f"Data shape: {df_attendance.shape}")
        print(f"Columns: {list(df_attendance.columns)}")
        print(f"Working directory: {csv_path}")
    except FileNotFoundError:
        print(f"Error: Could not find file {csv_path}")
        print(f"Current working directory files: {[f for f in __import__('os').listdir('.') if f.endswith('.csv')]}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # ---- Date prep (robust) ----
    if 'WEEK_NUMBER' not in df_attendance.columns and 'WEEK_BEGIN' not in df_attendance.columns:
        print("Error: Neither WEEK_BEGIN nor WEEK_NUMBER column found in data")
        return

    # Build/repair WEEK_BEGIN (internal only; output format unchanged)
    df_attendance = coerce_week_begin(df_attendance, week_col="WEEK_BEGIN", weeknum_col="WEEK_NUMBER")
    bad_rows = df_attendance['WEEK_BEGIN'].isna().sum()
    if bad_rows > 0:
        print(f"Dropping {bad_rows} rows with unusable WEEK_BEGIN (NaT) after coercion")
        df_attendance = df_attendance.dropna(subset=['WEEK_BEGIN'])

    # ---- Target column ----
    attendance_col = None
    if 'ATTENDANCE_RATE_WITH_OUTLIER' in df_attendance.columns:
        attendance_col = 'ATTENDANCE_RATE_WITH_OUTLIER'
    elif 'WEEKLY_ATTENDANCE_RATE' in df_attendance.columns:
        attendance_col = 'WEEKLY_ATTENDANCE_RATE'
    else:
        print("Error: No attendance rate column found in data")
        return

    # ---- Basic filter & de-dup (unchanged) ----
    df_attendance = df_attendance[df_attendance[attendance_col] < 130]

    key_columns = ['WEEK_NUMBER', 'WORK_LOCATION', 'SHIFT_TIME']
    if 'DEPARTMENT_GROUP' in df_attendance.columns:
        key_columns.append('DEPARTMENT_GROUP')
    available_key_columns = [col for col in key_columns if col in df_attendance.columns]
    df_attendance = df_attendance.drop_duplicates(subset=available_key_columns, keep='last')

    # Sort by week (internal)
    df_attendance = df_attendance.sort_values(['WEEK_BEGIN']).reset_index(drop=True)

    print(f"After preprocessing: {df_attendance.shape}")
    print(f"Date range: {df_attendance['WEEK_BEGIN'].min()} to {df_attendance['WEEK_BEGIN'].max()}")

    # ---- Grouping ----
    group_columns = ['WORK_LOCATION', 'SHIFT_TIME']
    if 'DEPARTMENT_GROUP' in df_attendance.columns:
        group_columns.append('DEPARTMENT_GROUP')

    all_results = []

    for group_values, df_group in df_attendance.groupby(group_columns):
        if len(group_columns) == 3:
            work_location, shift_time, department_group = group_values
        else:
            work_location, shift_time = group_values
            department_group = 'Unknown'

        print(f"\nProcessing: {work_location} - {shift_time} - {department_group}")
        print(f"Data points: {len(df_group)}")

        df_group = df_group.reset_index(drop=True)

        # --- Leakage-safe baseline column for later merge (same name as before) ---
        df_group['MOVING_AVG_4WEEK_FORECAST'] = calculate_moving_average(
            df_group, window=4, attendance_col=attendance_col, shift=2  # shift=2 keeps eval fair
        )

        # --- Greykite run ---
        greykite_results = run_greykite_realistic_forecast(
            df_group, actual_col=attendance_col, forecast_start_date='2024-01-01', gap_weeks=1
        )

        # --- Fallback if GK unavailable/empty ---
        if greykite_results is None or greykite_results.empty:
            print("  Using baseline fallback for this segment.")
            fb = baseline_realistic_forecast(
                df_group, actual_col=attendance_col, forecast_start_date='2024-01-01', gap_weeks=1, window=4
            )
            if fb is None or fb.empty:
                print("  No baseline could be produced (insufficient history before cutoffs). Skipping.")
                continue

            # Attach identifiers and WEEK_NUMBER for output parity
            fb['WORK_LOCATION'] = work_location
            fb['SHIFT_TIME'] = shift_time
            fb['DEPARTMENT_GROUP'] = department_group
            if 'WEEK_NUMBER' in df_group.columns:
                fb = fb.merge(df_group[['WEEK_BEGIN', 'WEEK_NUMBER']], on='WEEK_BEGIN', how='left')
            all_results.append(fb)
            continue

        # --- Merge GK with baseline + identifiers (preserve WEEK_NUMBER in output) ---
        greykite_results['WORK_LOCATION'] = work_location
        greykite_results['SHIFT_TIME'] = shift_time
        greykite_results['DEPARTMENT_GROUP'] = department_group

        merge_cols = ['WEEK_BEGIN', 'MOVING_AVG_4WEEK_FORECAST', attendance_col]
        if 'WEEK_NUMBER' in df_group.columns:
            merge_cols.append('WEEK_NUMBER')

        merged_results = greykite_results.merge(
            df_group[merge_cols],
            on='WEEK_BEGIN',
            how='left'
        ).rename(columns={attendance_col: 'ACTUAL_ATTENDANCE_RATE'})

        all_results.append(merged_results)

        # --- Metrics (same names) ---
        valid = merged_results.dropna(subset=['ACTUAL_ATTENDANCE_RATE', 'GREYKITE_FORECAST', 'MOVING_AVG_4WEEK_FORECAST'])
        if len(valid) > 0:
            mae_gk = mean_absolute_error(valid['ACTUAL_ATTENDANCE_RATE'], valid['GREYKITE_FORECAST'])
            mae_ma = mean_absolute_error(valid['ACTUAL_ATTENDANCE_RATE'], valid['MOVING_AVG_4WEEK_FORECAST'])
            print(f"  Greykite MAE: {mae_gk:.4f}")
            print(f"  Moving Avg MAE: {mae_ma:.4f}")

    # ---- Collate & export (SAME output columns/order as before) ----
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)

        output_columns = [
            'WEEK_BEGIN', 'WEEK_NUMBER', 'WORK_LOCATION', 'SHIFT_TIME', 'DEPARTMENT_GROUP',
            'ACTUAL_ATTENDANCE_RATE', 'GREYKITE_FORECAST', 'MOVING_AVG_4WEEK_FORECAST',
            'FOUR_WEEK_ROLLING_AVG', 'SIX_WEEK_ROLLING_AVG',
            'EXPONENTIAL_SMOOTHING_0_2', 'EXPONENTIAL_SMOOTHING_0_4',
            'EXPONENTIAL_SMOOTHING_0_6', 'EXPONENTIAL_SMOOTHING_0_8',
            'EXPONENTIAL_SMOOTHING_1',
            'TRAINING_WEEKS_USED', 'TRAINING_END_DATE', 'GAP_WEEKS'
        ]
        available_output_columns = [col for col in output_columns if col in final_results.columns]
        final_results = final_results[available_output_columns]

        sort_columns = ['WEEK_BEGIN', 'WORK_LOCATION', 'SHIFT_TIME']
        if 'DEPARTMENT_GROUP' in final_results.columns:
            sort_columns.append('DEPARTMENT_GROUP')
        final_results = final_results.sort_values(sort_columns)

        output_filename = f"attendance_forecast_results_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        final_results.to_csv(output_filename, index=False)

        print(f"\n=== RESULTS SUMMARY ===")
        print(f"Total records processed: {len(final_results)}")
        if {'WORK_LOCATION','SHIFT_TIME','DEPARTMENT_GROUP'}.issubset(final_results.columns):
            print(f"Unique combinations: {len(final_results.groupby(['WORK_LOCATION', 'SHIFT_TIME', 'DEPARTMENT_GROUP']))}")
        print(f"Date range: {final_results['WEEK_BEGIN'].min()} to {final_results['WEEK_BEGIN'].max()}")
        print(f"Results saved to: {output_filename}")

        print(f"\nSample results:")
        print(final_results.head(10).to_string(index=False))

        # Overall accuracy where both forecasts exist
        both_mask = final_results[['ACTUAL_ATTENDANCE_RATE','GREYKITE_FORECAST','MOVING_AVG_4WEEK_FORECAST']].notna().all(axis=1)
        valid_final = final_results.loc[both_mask]
        if len(valid_final) > 0:
            overall_mae_gk = mean_absolute_error(valid_final['ACTUAL_ATTENDANCE_RATE'], valid_final['GREYKITE_FORECAST'])
            overall_mae_ma = mean_absolute_error(valid_final['ACTUAL_ATTENDANCE_RATE'], valid_final['MOVING_AVG_4WEEK_FORECAST'])
            print(f"\n=== OVERALL ACCURACY (where both exist) ===")
            print(f"Overall Greykite MAE: {overall_mae_gk:.4f}")
            print(f"Overall Moving Average MAE: {overall_mae_ma:.4f}")
            if overall_mae_gk < overall_mae_ma:
                print("Greykite performs better overall")
            else:
                print("Moving Average performs better overall")
    else:
        print("No results generated. Please check your data and requirements.")

if __name__ == "__main__":
    main()
