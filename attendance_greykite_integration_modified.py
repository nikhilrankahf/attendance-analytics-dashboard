#!/usr/bin/env python3
"""
Enhanced Attendance Forecasting Script using Greykite
- Keeps the OLD CSV SCHEMA in the output:
  week_number, work_location, shift_time, department_group,
  actual_attendance, greykite_forecast, [greykite_forecast_lower_95, greykite_forecast_upper_95],
  plus baseline columns (lowercase) from input if present, and four_week_rolling_avg_shift2.
- Retains new logic:
  * T+2 horizon with 1-week blackout (gap=1)
  * Leakage-safe baseline MA(4) shifted by 2
  * AR lags [1,2,3,4,52] + post-holiday effects + winsorization
  * Robust date handling from WEEK_NUMBER/WEEK_BEGIN
  * Greykite fallback baseline (leakage-safe) when GK can't run
- Uses Thursday week start alignment (W-THU) to match historical convention
"""

import pandas as pd
import numpy as np
import datetime as dt
import warnings
import argparse
from sklearn.metrics import mean_absolute_error

# Greykite imports
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.templates.autogen.forecast_config import ComputationParam
from greykite.framework.templates.autogen.forecast_config import (
    ForecastConfig, MetadataParam, ModelComponentsParam, EvaluationMetricParam, EvaluationPeriodParam
)

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', None)

# ---------------------------
# HelloFresh week helpers (unchanged signatures)
# ---------------------------
def datetime_to_hf_week(date_time: dt.datetime):
    hf_datetime = date_time + dt.timedelta(days=2)
    year, week, _ = hf_datetime.isocalendar()
    return f"{year}-W{week:02d}"

def hf_week_to_datetime(hf_week: str):
    # ISO week Monday=1; we want HF Thursday anchor overall; we still compute base date then align via freq later
    return dt.datetime.strptime(hf_week + "-1", "%G-W%V-%w") - dt.timedelta(days=2)

def hf_week_add(hf_week: str, num_weeks: int):
    return datetime_to_hf_week(hf_week_to_datetime(hf_week) + dt.timedelta(weeks=num_weeks))

def hf_week_sub(hf_week: str, num_weeks: int):
    return datetime_to_hf_week(hf_week_to_datetime(hf_week) - dt.timedelta(weeks=num_weeks))

def current_hf_week():
    return datetime_to_hf_week(dt.datetime.today())

def week_to_thursday_date(week_str: str) -> pd.Timestamp:
    """Convert 'YYYY-Www' to that ISO week's Thursday date (day=4)."""
    year, week = map(int, week_str.split("-W"))
    thursday = dt.date.fromisocalendar(year, week, 4)
    return pd.Timestamp(thursday)

# ---------------------------
# Robust week/date utilities (internal only; output schema unaffected)
# ---------------------------
def safe_week_str(ts: pd.Timestamp, fmt: str = "%G-W%V"):
    try:
        return ts.strftime(fmt) if pd.notna(ts) else "NaT"
    except Exception:
        return "NaT"

def coerce_week_begin(df: pd.DataFrame,
                      week_col: str = "WEEK_BEGIN",
                      weeknum_col: str = "WEEK_NUMBER") -> pd.DataFrame:
    """
    Build/repair WEEK_BEGIN:
      - Parse WEEK_BEGIN if present
      - If NaT and WEEK_NUMBER exists, rebuild from HF week string (Thursday)
    """
    if week_col in df.columns:
        df[week_col] = pd.to_datetime(df[week_col], errors="coerce")
    else:
        df[week_col] = pd.NaT

    if weeknum_col in df.columns:
        mask = df[week_col].isna() & df[weeknum_col].notna()
        if mask.any():
            df.loc[mask, week_col] = df.loc[mask, weeknum_col].apply(week_to_thursday_date)

    return df

# ---------------------------
# Baseline calc (leakage-safe). We’ll export as 'four_week_rolling_avg_shift2'
# ---------------------------
def leakage_safe_ma4_shift2(series: pd.Series) -> pd.Series:
    """MA(4) shifted by 2 for T+2 with 1-week blackout."""
    return series.rolling(window=4, min_periods=1).mean().shift(2)

# ---------------------------
# Greykite realistic forecast (T+2 with gap=1)
# Returns a DataFrame with rows per target week (WEEK_BEGIN, GREYKITE_FORECAST, intervals if available, etc.)
# ---------------------------
# ---------------------------
# Greykite realistic forecast (T+2 with gap=1) — uses 'y' as modeling target
# ---------------------------
def run_greykite_realistic_forecast(
    df_group: pd.DataFrame,
    actual_col: str,
    forecast_start_date: str = "2024-01-01",
    gap_weeks: int = 1
) -> pd.DataFrame:
    """
    Rolling-origin Greykite forecast:
      - For each target week >= forecast_start_date
      - Train cutoff = target - (gap+1) weeks (so with gap=1, cutoff = T-2)
      - Forecast horizon = gap+1 (=2), take step at index 'gap_weeks' (1) => T+2
    """
    MIN_TRAIN_WEEKS = 52

    # Robust prep
    df_group = (
        df_group
        .dropna(subset=["WEEK_BEGIN"])
        .sort_values("WEEK_BEGIN")
        .reset_index(drop=True)
        .copy()
    )
    if len(df_group) < MIN_TRAIN_WEEKS:
        print(f"    Skipping Greykite: insufficient total history ({len(df_group)} < {MIN_TRAIN_WEEKS})")
        return pd.DataFrame()

    # --- Standardize modeling target to 'y' (do NOT overwrite original actuals) ---
    df_group["y"] = pd.to_numeric(df_group[actual_col], errors="coerce")

    # Winsorize the modeling target y for robustness (adjust thresholds if needed)
    lo, hi = df_group["y"].quantile([0.005, 0.995])
    df_group["y"] = df_group["y"].clip(lo, hi)

    # Leakage-safe rolling features (optional regressors later; no leakage because of shift)
    df_group["ROLL_MEAN_4_S2"]   = df_group["y"].rolling(4, min_periods=1).mean().shift(2)
    df_group["ROLL_MEDIAN_4_S2"] = df_group["y"].rolling(4, min_periods=1).median().shift(2)

    forecast_start = pd.to_datetime(forecast_start_date)
    forecast_weeks = df_group[df_group["WEEK_BEGIN"] >= forecast_start].copy()
    if forecast_weeks.empty:
        print(f"    No data found from {forecast_start_date} onwards")
        return pd.DataFrame()

    results_rows = []
    forecaster = Forecaster()

    for _, row in forecast_weeks.iterrows():
        target_week = row["WEEK_BEGIN"]
        actual_value = row[actual_col]

        # Training cutoff T-2 (gap=1)
        training_cutoff = target_week - pd.Timedelta(weeks=gap_weeks + 1)
        train_data = df_group[df_group["WEEK_BEGIN"] <= training_cutoff].copy()
        train_data = train_data.dropna(subset=["WEEK_BEGIN", "y"])

        if len(train_data) < MIN_TRAIN_WEEKS:
            print(f"    Skip target {safe_week_str(target_week)}: train weeks {len(train_data)} < {MIN_TRAIN_WEEKS}")
            continue

        print(
            f"    Greykite for {safe_week_str(target_week)} using {len(train_data)} weeks up to "
            f"{safe_week_str(train_data['WEEK_BEGIN'].max())} (gap={gap_weeks})"
        )

        # Greykite metadata: point the model to 'y'
        metadata = MetadataParam(
            time_col="WEEK_BEGIN",
            value_col="y",     # <-- use standardized modeling target
            freq="W-THU"
        )

        # Model components (AR kept on with version-safe schema)
        model_components = ModelComponentsParam(
            seasonality={
                "yearly_seasonality": 4,
                "weekly_seasonality": False,
                "monthly_seasonality": False,
                "daily_seasonality": False,
                "quarterly_seasonality": False,
            },
            autoregression={
                "autoreg_dict": {
                    "lag_dict": {                  # correct key
                        "orders": [1, 2, 3, 4, 52] # drop 52 if your install complains
                    }
                }
            },
            events={
                "holiday_lookup_countries": ["US"],
                "holiday_pre_num_days": 8,
                "holiday_post_num_days": 3
            }
        )

        evaluation = EvaluationPeriodParam(
            test_horizon=2,
            periods_between_train_test=1,
            cv_horizon=2,
            cv_min_train_periods=MIN_TRAIN_WEEKS,
            cv_use_most_recent_splits=True,
            cv_max_splits=0   # no CV inside the rolling loop
        )

        try:
            result = forecaster.run_forecast_config(
                df=train_data,
                config=ForecastConfig(
                    model_template=ModelTemplateEnum.SILVERKITE.name,
                    forecast_horizon=gap_weeks + 1,   # (=2), take index 1
                    coverage=0.95,                    # keep if you want intervals; set None for speed
                    metadata_param=metadata,
                    model_components_param=model_components,
                    evaluation_metric_param=EvaluationMetricParam(
                        cv_selection_metric="MeanAbsoluteError"  # switch to sMAPE if you prefer APE-like selection
                    ),
                    evaluation_period_param=evaluation,
                    computation_param=ComputationParam(n_jobs=-1)
                )
            )

            fdf = result.forecast.df
            if len(fdf) > gap_weeks:
                # Take the T+2 point (index = gap_weeks)
                rowf = fdf.iloc[gap_weeks]
                forecast_value = rowf.get("forecast", np.nan)
                lower_value = rowf.get("forecast_lower", np.nan) if "forecast_lower" in fdf.columns else np.nan
                upper_value = rowf.get("forecast_upper", np.nan) if "forecast_upper" in fdf.columns else np.nan

                # --- Tiny bias correction using last 12 in-sample residuals on 'y' ---
                tail = train_data.tail(12).copy()
                # Predict on the last 12 training timestamps
                ins = result.model.predict(tail[["WEEK_BEGIN"]].rename(columns={"WEEK_BEGIN": "ts"}))
                resid = tail["y"].to_numpy() - ins["forecast"].to_numpy()
                bias = np.nanmean(resid)
                if not np.isnan(bias):
                    forecast_value = forecast_value + bias

                results_rows.append({
                    "WEEK_BEGIN": target_week,
                    "ACTUAL_ATTENDANCE_RATE": actual_value,   # keep original actuals for output mapping
                    "GREYKITE_FORECAST": forecast_value,
                    "GREYKITE_FORECAST_LOWER_95": lower_value,
                    "GREYKITE_FORECAST_UPPER_95": upper_value,
                    "TRAINING_WEEKS_USED": len(train_data),
                    "TRAINING_END_DATE": train_data["WEEK_BEGIN"].max(),
                    "GAP_WEEKS": gap_weeks
                })
        except Exception as e:
            print(f"      Greykite error at {safe_week_str(target_week)}: {e}")
            continue

    return pd.DataFrame(results_rows)

# ---------------------------
# Baseline realistic forecast (fallback); uses leakage-safe MA(4) shift=2
# ---------------------------
def baseline_realistic_forecast(
    df_group: pd.DataFrame,
    actual_col: str,
    forecast_start_date: str = "2024-01-01",
    gap_weeks: int = 1,
    window: int = 4
) -> pd.DataFrame:
    """
    Leakage-safe baseline to mirror realistic setup:
      - For target_week, training cutoff = target - (gap+1) weeks
      - Baseline MA(4) shifted by 2 already respects cutoff (no leakage)
    """
    df_group = df_group.sort_values("WEEK_BEGIN").reset_index(drop=True)
    forecast_start = pd.to_datetime(forecast_start_date)

    df_group["__BASELINE_MA4_S2"] = df_group[actual_col].rolling(window, min_periods=1).mean().shift(gap_weeks + 1)

    target_rows = df_group[df_group["WEEK_BEGIN"] >= forecast_start].copy()
    if target_rows.empty:
        return pd.DataFrame()

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
            "GREYKITE_FORECAST_LOWER_95": np.nan,
            "GREYKITE_FORECAST_UPPER_95": np.nan,
            "BASELINE_MA4_S2": r["__BASELINE_MA4_S2"],
            "TRAINING_WEEKS_USED": len(train_slice),
            "TRAINING_END_DATE": train_slice["WEEK_BEGIN"].max(),
            "GAP_WEEKS": gap_weeks
        })

    return pd.DataFrame(rows)

# ---------------------------
# Assemble rows in the OLD CSV SCHEMA (lowercase; week_number as key)
# ---------------------------
def assemble_output_rows(
    df_group: pd.DataFrame,                   # original group df (has WEEK_NUMBER, actuals, and possibly baseline cols)
    group_id: dict,                           # {'WORK_LOCATION', 'SHIFT_TIME', 'DEPARTMENT_GROUP'}
    gk_df: pd.DataFrame = None,               # Greykite per-target rows (can be empty)
    fb_df: pd.DataFrame = None,               # fallback per-target rows (can be empty)
    attendance_col: str = "WEEKLY_ATTENDANCE_RATE",
    include_years=("2024", "2025")
) -> list:
    """
    Emits rows in the old schema:
      week_number, work_location, shift_time, department_group,
      actual_attendance, greykite_forecast, [greykite_forecast_lower_95, greykite_forecast_upper_95],
      + baseline columns (lowercase) from input if present,
      + four_week_rolling_avg_shift2 (our leakage-safe MA4 baseline).
    """
    out = []

    # Ensure WEEK_NUMBER exists (use existing or reconstruct)
    if "WEEK_NUMBER" not in df_group.columns:
        if "WEEK_BEGIN" in df_group.columns:
            df_group["WEEK_NUMBER"] = df_group["WEEK_BEGIN"].apply(lambda d: d.strftime("%G-W%V") if pd.notna(d) else None)
        else:
            df_group["WEEK_NUMBER"] = None

    # lowercase grouping identifiers
    gl = {
        "work_location": group_id.get("WORK_LOCATION", None),
        "shift_time": group_id.get("SHIFT_TIME", None),
        "department_group": group_id.get("DEPARTMENT_GROUP", None)
    }

    # Map WEEK_BEGIN -> WEEK_NUMBER
    wb_to_wn = {}
    if "WEEK_BEGIN" in df_group.columns:
        tmp = df_group[["WEEK_BEGIN", "WEEK_NUMBER"]].dropna().drop_duplicates()
        wb_to_wn = dict(zip(tmp["WEEK_BEGIN"], tmp["WEEK_NUMBER"]))

    # Add our leakage-safe baseline as lowercase name for output
    if "MOVING_AVG_4WEEK_FORECAST" in df_group.columns:
        df_group["four_week_rolling_avg_shift2"] = df_group["MOVING_AVG_4WEEK_FORECAST"]
    elif "BASELINE_MA4_S2" in df_group.columns:
        df_group["four_week_rolling_avg_shift2"] = df_group["BASELINE_MA4_S2"]

    # Candidate baseline columns from input (old names)
    other_forecast_columns = [
        "FOUR_WEEK_ROLLING_AVG",
        "SIX_WEEK_ROLLING_AVG",
        "EXPONENTIAL_SMOOTHING_0_2",
        "EXPONENTIAL_SMOOTHING_0_4",
        "EXPONENTIAL_SMOOTHING_0_6",
        "EXPONENTIAL_SMOOTHING_0_8",
        "EXPONENTIAL_SMOOTHING_1"
    ]

    df_group_years = df_group[df_group["WEEK_NUMBER"].astype(str).str[:4].isin(include_years)]

    def attach_baselines(row_dict, src_row: pd.Series):
        for col in other_forecast_columns:
            if col in src_row.index:
                if col.startswith("EXPONENTIAL_SMOOTHING_"):
                    alpha = col.split("_")[-1].replace("_", ".")
                    key = f"exponential_smoothing_alpha_{alpha}"
                else:
                    key = col.lower()
                row_dict[key] = src_row[col]
        if "four_week_rolling_avg_shift2" in src_row.index:
            row_dict["four_week_rolling_avg_shift2"] = src_row["four_week_rolling_avg_shift2"]

    # 1) Greykite rows
    if gk_df is not None and not gk_df.empty:
        for _, r in gk_df.iterrows():
            wb = r["WEEK_BEGIN"]
            wn = wb_to_wn.get(wb, (wb.strftime("%G-W%V") if pd.notna(wb) else None))
            if not (isinstance(wn, str) and wn[:4] in include_years):
                continue

            src = df_group_years[df_group_years["WEEK_NUMBER"] == wn]
            src_row = src.iloc[0] if not src.empty else pd.Series(dtype=object)
            actual = src_row.get(attendance_col, np.nan) if not src.empty else np.nan

            row = {
                "week_number": wn,
                "work_location": gl["work_location"],
                "shift_time": gl["shift_time"],
                "department_group": gl["department_group"],
                "actual_attendance": actual,
                "greykite_forecast": r.get("GREYKITE_FORECAST", np.nan)
            }

            # intervals if available
            lo = r.get("GREYKITE_FORECAST_LOWER_95", np.nan)
            hi = r.get("GREYKITE_FORECAST_UPPER_95", np.nan)
            if not pd.isna(lo) and not pd.isna(hi):
                row["greykite_forecast_lower_95"] = lo
                row["greykite_forecast_upper_95"] = hi

            attach_baselines(row, src_row)
            out.append(row)

    # 2) Fallback rows (no Greykite point)
    if fb_df is not None and not fb_df.empty:
        for _, r in fb_df.iterrows():
            wb = r["WEEK_BEGIN"]
            wn = wb_to_wn.get(wb, (wb.strftime("%G-W%V") if pd.notna(wb) else None))
            if not (isinstance(wn, str) and wn[:4] in include_years):
                continue

            src = df_group_years[df_group_years["WEEK_NUMBER"] == wn]
            src_row = src.iloc[0] if not src.empty else pd.Series(dtype=object)
            actual = src_row.get(attendance_col, np.nan) if not src.empty else np.nan

            row = {
                "week_number": wn,
                "work_location": gl["work_location"],
                "shift_time": gl["shift_time"],
                "department_group": gl["department_group"],
                "actual_attendance": actual,
                "greykite_forecast": np.nan  # fallback provides no GK point
            }
            attach_baselines(row, src_row)
            out.append(row)

    return out

# ---------------------------
# Main
# ---------------------------
def main():
    print("=== ENHANCED ATTENDANCE FORECASTING WITH GREYKITE (OLD CSV SCHEMA) ===")
    
    # CLI filters
    parser = argparse.ArgumentParser(description="Enhanced Attendance Forecasting with optional filters")
    parser.add_argument("--work_location", type=str, default=None, help="Filter to a single WORK_LOCATION")
    parser.add_argument("--shift_time", type=str, default=None, help="Filter to a single SHIFT_TIME")
    parser.add_argument("--department_group", type=str, default=None, help="Filter to a single DEPARTMENT_GROUP")
    parser.add_argument("--min_week", type=str, default=None, help="Minimum target week (YYYY-Www) for forecasting, e.g., 2025-W28")
    args = parser.parse_args()
    
    # Load data
    input_file = "/Users/nikhil.ranka/attendance-analytics-dashboard/Labor_Management-Greykite_Input.csv"
    try:
        df = pd.read_csv(input_file)
        print(f"\n✓ Loaded data: {len(df)} records from {input_file}")
        print(f"Available columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"✗ Error: Could not find {input_file}")
        return
    
    # Identify actuals column
    preferred_actual_cols = [
        "WEEKLY_ATTENDANCE_RATE",
        "Weekly_attendance_rate",
        "ATTENDANCE_RATE_WITH_OUTLIER"
    ]
    attendance_col = None
    for col in preferred_actual_cols:
        if col in df.columns:
            attendance_col = col
            break
    if attendance_col is None:
        print("✗ Error: Could not find an attendance rate column. Expected one of:", preferred_actual_cols)
        return
    print(f"Using actuals column: '{attendance_col}'")

    # Build/repair WEEK_BEGIN internally (not exported)
    if "WEEK_NUMBER" not in df.columns and "WEEK_BEGIN" not in df.columns:
        print("✗ Error: Neither WEEK_BEGIN nor WEEK_NUMBER column found in data")
        return
    df = coerce_week_begin(df, week_col="WEEK_BEGIN", weeknum_col="WEEK_NUMBER")
    bad_rows = df["WEEK_BEGIN"].isna().sum()
    if bad_rows > 0:
        print(f"Dropping {bad_rows} rows with unusable WEEK_BEGIN (NaT) after coercion")
        df = df.dropna(subset=["WEEK_BEGIN"])

    # Light hygiene: cap ridiculous values; de-dup
    df = df[df[attendance_col] < 130]
    key_columns = ["WEEK_NUMBER", "WORK_LOCATION", "SHIFT_TIME"]
    if "DEPARTMENT_GROUP" in df.columns:
        key_columns.append("DEPARTMENT_GROUP")
    available_key_columns = [c for c in key_columns if c in df.columns]
    df = df.drop_duplicates(subset=available_key_columns, keep="last")
    df = df.sort_values(["WEEK_BEGIN"]).reset_index(drop=True)

    # Apply segment filters (do NOT filter by min_week here; keep history for training)
    if any([args.work_location, args.shift_time, args.department_group]):
        print("\nApplying segment filters:")
        if args.work_location is not None:
            df = df[df["WORK_LOCATION"] == args.work_location]
            print(f"  WORK_LOCATION == {args.work_location} -> {len(df)} rows")
        if args.shift_time is not None and "SHIFT_TIME" in df.columns:
            df = df[df["SHIFT_TIME"] == args.shift_time]
            print(f"  SHIFT_TIME == {args.shift_time} -> {len(df)} rows")
        if args.department_group is not None and "DEPARTMENT_GROUP" in df.columns:
            df = df[df["DEPARTMENT_GROUP"] == args.department_group]
            print(f"  DEPARTMENT_GROUP == {args.department_group} -> {len(df)} rows")
        if len(df) == 0:
            print("✗ After applying filters, no data remains. Exiting.")
            return

    # Determine forecast start date from min_week if provided (targets only; training can use prior weeks)
    forecast_start_date = "2024-01-01"
    if args.min_week is not None:
        try:
            min_week_ts = week_to_thursday_date(args.min_week)
            forecast_start_date = min_week_ts.strftime("%Y-%m-%d")
            print(f"\nForecast targets constrained to weeks >= {args.min_week} (start date {forecast_start_date})")
        except Exception as e:
            print(f"! Warning: Could not parse --min_week '{args.min_week}': {e}. Using default {forecast_start_date}")

    # Grouping columns
    grouping_cols = ["WORK_LOCATION", "SHIFT_TIME", "DEPARTMENT_GROUP"]
    available_cols = [c for c in grouping_cols if c in df.columns]
    if not available_cols:
        print("✗ Error: Required grouping columns not found in data")
        return
    print(f"\nGrouping by: {available_cols}")
    
    all_output_rows = []
    successful_segments = 0

    for group_values, df_group in df.groupby(available_cols):
        group_id = dict(zip(available_cols, group_values))
        # Ensure all keys exist for downstream assembly
        for k in ["WORK_LOCATION", "SHIFT_TIME", "DEPARTMENT_GROUP"]:
            group_id.setdefault(k, None)

        print(f"\nProcessing: {group_id['WORK_LOCATION']} - {group_id['SHIFT_TIME']} - {group_id['DEPARTMENT_GROUP']}")
        print(f"Data points: {len(df_group)}")

        df_group = df_group.reset_index(drop=True)

        # Leakage-safe baseline column for downstream export and eval (lowercase in final)
        df_group["MOVING_AVG_4WEEK_FORECAST"] = leakage_safe_ma4_shift2(df_group[attendance_col])

        # Try Greykite per-target T+2
        gk_df = run_greykite_realistic_forecast(
            df_group=df_group,
            actual_col=attendance_col,
            forecast_start_date=forecast_start_date,
            gap_weeks=1
        )

        # Fallback if Greykite unavailable/empty
        fb_df = pd.DataFrame()
        if gk_df is None or gk_df.empty:
            print("  Using baseline fallback for this segment.")
            fb_df = baseline_realistic_forecast(
                df_group=df_group,
                actual_col=attendance_col,
                forecast_start_date=forecast_start_date,
                gap_weeks=1,
                window=4
            )
            if fb_df is None or fb_df.empty:
                print("  No baseline could be produced (insufficient history before cutoffs). Skipping segment.")
                continue
        
        # Assemble rows in old schema (lowercase + week_number)
        segment_rows = assemble_output_rows(
            df_group=df_group,
            group_id=group_id,
            gk_df=gk_df,
            fb_df=fb_df,
            attendance_col=attendance_col,
            include_years=("2024", "2025")  # mirror old file scope
        )
        if segment_rows:
            all_output_rows.extend(segment_rows)
            successful_segments += 1

        # Quick on-the-fly MAE check (only where both GK point and actual exist)
        if gk_df is not None and not gk_df.empty:
            # link to actuals & baseline
            tmp = gk_df.merge(
                df_group[["WEEK_BEGIN", attendance_col, "MOVING_AVG_4WEEK_FORECAST"]],
                on="WEEK_BEGIN", how="left"
            ).rename(columns={attendance_col: "ACTUAL"})
            valid = tmp.dropna(subset=["ACTUAL", "GREYKITE_FORECAST", "MOVING_AVG_4WEEK_FORECAST"])
            if len(valid) > 0:
                mae_gk = mean_absolute_error(valid["ACTUAL"], valid["GREYKITE_FORECAST"])
                mae_ma = mean_absolute_error(valid["ACTUAL"], valid["MOVING_AVG_4WEEK_FORECAST"])
                print(f"  Greykite MAE: {mae_gk:.4f} | Shifted MA(4) MAE: {mae_ma:.4f}")

    # Export EXACTLY like the old script
    if all_output_rows:
        output_df = pd.DataFrame(all_output_rows)

        # Old filename pattern
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"enhanced_attendance_forecast_all_combinations_2024_2025_{timestamp}.csv"
        
        # Core old columns in order
        core_cols = [
            "week_number", "work_location", "shift_time", "department_group",
            "actual_attendance", "greykite_forecast"
        ]
        # Optional intervals
        if {"greykite_forecast_lower_95", "greykite_forecast_upper_95"}.issubset(set(output_df.columns)):
            core_cols += ["greykite_forecast_lower_95", "greykite_forecast_upper_95"]

        # Keep any other baseline columns (already lowercase)
        other_cols = [c for c in output_df.columns if c not in core_cols]
        output_df = output_df[core_cols + other_cols]

        # Save
        output_df.to_csv(output_file, index=False)
        print(f"\n✓ Enhanced forecasts saved to: {output_file}")
        print(f"  Records: {len(output_df)}")
        print(f"  Columns: {list(output_df.columns)}")
        print(f"  Segments processed successfully: {successful_segments}")
    else:
        print("\n✗ No forecasts generated - no segments had sufficient usable data")
    
    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
