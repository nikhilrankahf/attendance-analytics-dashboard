# --- after your imports and helper defs, keep all the improved logic as we set up before ---
# (Greykite config with T+2, gap=1, AR lags, post-holiday, winsorization, date hardening, baseline fallback, etc.)
# Only the lines below change the *extraction* and *export* to match the old CSV schema.

def extract_greykite_point(result, step_index):
    """Pull point + interval for the desired step from Greykite forecast df."""
    fdf = result.forecast.df
    if len(fdf) <= step_index:
        return None, None, None
    fc = fdf.iloc[step_index]
    yhat = fc.get("forecast", np.nan)
    lo = fc.get("forecast_lower", np.nan) if "forecast_lower" in fdf.columns else np.nan
    hi = fc.get("forecast_upper", np.nan) if "forecast_upper" in fdf.columns else np.nan
    return yhat, lo, hi

def assemble_output_rows(
    df_group,                       # original group df (has WEEK_NUMBER, actuals, any baseline cols)
    group_id,                       # dict with {'WORK_LOCATION', 'SHIFT_TIME', 'DEPARTMENT_GROUP'}
    gk_rows_df=None,                # dataframe from run_greykite_realistic_forecast (per-target rows)
    gk_intervals=None,              # dict: {week_begin -> (yhat, lo, hi)} when available
    fallback_rows_df=None,          # dataframe from baseline_realistic_forecast (when GK missing)
    attendance_col="WEEKLY_ATTENDANCE_RATE",
    include_years=("2024","2025")   # to mirror the older file scope, filter to these years
):
    """
    Returns a list of dicts in the *old schema*:
      week_number, work_location, shift_time, department_group,
      actual_attendance, greykite_forecast, greykite_forecast_lower_95, greykite_forecast_upper_95,
      plus baseline columns if present in the input (lowercase names).
      Also adds our leakage-safe MA(4) as 'four_week_rolling_avg_shift2' for clarity.
    """
    out = []

    # Prepare convenience lookups
    df_group = df_group.copy()
    # old schema uses WEEK_NUMBER string; keep it
    if "WEEK_NUMBER" not in df_group.columns:
        # if missing, reconstruct from WEEK_BEGIN (internal) if available
        if "WEEK_BEGIN" in df_group.columns:
            df_group["WEEK_NUMBER"] = df_group["WEEK_BEGIN"].apply(
                lambda d: d.strftime("%G-W%V") if pd.notna(d) else None
            )
        else:
            df_group["WEEK_NUMBER"] = None

    # lowercase grouping values
    gl = {
        "work_location": group_id.get("WORK_LOCATION", None),
        "shift_time": group_id.get("SHIFT_TIME", None),
        "department_group": group_id.get("DEPARTMENT_GROUP", None)
    }

    # candidate baseline columns from input (old names)
    other_forecast_columns = [
        "FOUR_WEEK_ROLLING_AVG",
        "SIX_WEEK_ROLLING_AVG",
        "EXPONENTIAL_SMOOTHING_0_2",
        "EXPONENTIAL_SMOOTHING_0_4",
        "EXPONENTIAL_SMOOTHING_0_6",
        "EXPONENTIAL_SMOOTHING_0_8",
        "EXPONENTIAL_SMOOTHING_1"
    ]
    # leakage-safe MA(4) we built as MOVING_AVG_4WEEK_FORECAST (shift=2)
    if "MOVING_AVG_4WEEK_FORECAST" in df_group.columns:
        df_group["four_week_rolling_avg_shift2"] = df_group["MOVING_AVG_4WEEK_FORECAST"]

    # restrict to target years like the old script
    df_years = df_group[df_group["WEEK_NUMBER"].astype(str).str[:4].isin(include_years)]

    # helper to create a row in old schema
    def make_row(wn, actual, gk, lo, hi, src_row):
        row = {
            "week_number": wn,
            "work_location": gl["work_location"],
            "shift_time": gl["shift_time"],
            "department_group": gl["department_group"],
            "actual_attendance": actual,
            "greykite_forecast": gk
        }
        # optional intervals
        if not pd.isna(lo):
            row["greykite_forecast_lower_95"] = lo
            row["greykite_forecast_upper_95"] = hi

        # attach any known baseline columns from *input* (converted to readable lowercase)
        for col in other_forecast_columns:
            if col in src_row.index:
                if col.startswith("EXPONENTIAL_SMOOTHING_"):
                    # map e.g. EXPONENTIAL_SMOOTHING_0_4 -> exponential_smoothing_alpha_0.4
                    alpha = col.split("_")[-1].replace("_", ".")
                    key = f"exponential_smoothing_alpha_{alpha}"
                else:
                    key = col.lower()
                row[key] = src_row[col]

        # attach our leakage-safe baseline if present
        if "four_week_rolling_avg_shift2" in src_row.index:
            row["four_week_rolling_avg_shift2"] = src_row["four_week_rolling_avg_shift2"]

        return row

    # Build a map week_begin -> WEEK_NUMBER for linking GK/fallback rows back to week_number
    wb_to_wn = {}
    if "WEEK_BEGIN" in df_group.columns:
        tmp = df_group[["WEEK_BEGIN", "WEEK_NUMBER"]].dropna().drop_duplicates()
        wb_to_wn = dict(zip(tmp["WEEK_BEGIN"], tmp["WEEK_NUMBER"]))

    # 1) Greykite rows first (preferred when present)
    if gk_rows_df is not None and not gk_rows_df.empty:
        for _, r in gk_rows_df.iterrows():
            wb = r["WEEK_BEGIN"]
            wn = wb_to_wn.get(wb, None)
            if wn is None:
                # fallback: derive week number string from WEEK_BEGIN
                wn = wb.strftime("%G-W%V") if pd.notna(wb) else None
            # filter to target years like old file
            if not (isinstance(wn, str) and wn[:4] in include_years):
                continue

            # locate the source row to harvest actuals & baselines
            src = df_years[df_years["WEEK_NUMBER"] == wn]
            src_row = src.iloc[0] if not src.empty else pd.Series(dtype=object)

            actual = src_row.get(attendance_col, np.nan) if not src.empty else np.nan

            # pick up forecast + intervals
            if gk_intervals and wb in gk_intervals:
                gk, lo, hi = gk_intervals[wb]
            else:
                # if intervals not captured earlier, at least set point
                gk = r.get("GREYKITE_FORECAST", np.nan)
                lo = np.nan
                hi = np.nan

            out.append(make_row(wn, actual, gk, lo, hi, src_row))

    # 2) Fallback rows for weeks not covered by GK
    if fallback_rows_df is not None and not fallback_rows_df.empty:
        for _, r in fallback_rows_df.iterrows():
            wb = r["WEEK_BEGIN"]
            wn = wb_to_wn.get(wb, None)
            if wn is None:
                wn = wb.strftime("%G-W%V") if pd.notna(wb) else None
            if not (isinstance(wn, str) and wn[:4] in include_years):
                continue

            src = df_years[df_years["WEEK_NUMBER"] == wn]
            src_row = src.iloc[0] if not src.empty else pd.Series(dtype=object)
            actual = src_row.get(attendance_col, np.nan) if not src.empty else np.nan

            # baseline fallback has no GK, so greykite_forecast stays NaN
            out.append(make_row(wn, actual, np.nan, np.nan, np.nan, src_row))

    return out

# ---- In your main() loop, after you run Greykite / fallback per segment, replace the export block with: ----

# inside the per-group loop, collect rows in old schema
segment_rows = []

# After Greykite run:
# We also capture intervals per target week_begin
gk_intervals_map = {}
if greykite_results is not None and not greykite_results.empty:
    # If you have access to the 'result' object per target, you can stash (yhat, lo, hi) there.
    # In our rolling realistic loop we only kept point yhat. Quick way:
    # recompute intervals from the stored forecast_df if you still have the `result`.
    # If not, leave intervals as NaN (schema still matches).
    pass

segment_rows.extend(
    assemble_output_rows(
        df_group=df_group,
        group_id={"WORK_LOCATION": work_location, "SHIFT_TIME": shift_time, "DEPARTMENT_GROUP": department_group},
        gk_rows_df=greykite_results,              # dataframe of GK targets (has WEEK_BEGIN, GREYKITE_FORECAST, etc.)
        gk_intervals=gk_intervals_map,            # if you recorded intervals; else leave {}
        fallback_rows_df=fb if ('fb' in locals() and fb is not None) else None,
        attendance_col=attendance_col,
        include_years=("2024","2025")             # mirror older file scope
    )
)

all_output_data.extend(segment_rows)

# ---- After processing all groups, export EXACTLY like the old script: ----

if all_output_data:
    output_df = pd.DataFrame(all_output_data)

    # filename pattern identical to the old file
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"enhanced_attendance_forecast_all_combinations_2024_2025_{timestamp}.csv"

    # enforce column order where possible (old schema core)
    core_cols = [
        "week_number", "work_location", "shift_time", "department_group",
        "actual_attendance", "greykite_forecast"
    ]
    # optional intervals
    if {"greykite_forecast_lower_95", "greykite_forecast_upper_95"}.issubset(output_df.columns):
        core_cols += ["greykite_forecast_lower_95", "greykite_forecast_upper_95"]

    # keep any other baseline columns that happen to be present (already lowercase in assemble_output_rows)
    other_cols = [c for c in output_df.columns if c not in core_cols]
    output_df = output_df[core_cols + other_cols]

    output_df.to_csv(output_file, index=False)
    print(f"\n✓ Enhanced forecasts saved to: {output_file}")
    print(f"  Records: {len(output_df)}")
    print(f"  Columns: {list(output_df.columns)}")
else:
    print("\n✗ No forecasts generated - no combinations had sufficient data")
