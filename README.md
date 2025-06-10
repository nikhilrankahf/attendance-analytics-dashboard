# 🎯 Advanced Attendance Analytics Dashboard

A comprehensive Streamlit dashboard for analyzing Greykite vs Moving Average forecast performance with **MAPE-focused analysis**.

## 📊 Features

- **Executive Performance Summary**: Key insights and trends with MAPE highlights
- **Enhanced KPI Metrics**: Win rates, MAPE improvements, and performance indicators
- **MAPE-Focused Outliers Analysis**: Segment-based MAPE analysis instead of individual APE values
- **Performance Timeline**: MAPE trends over time with comprehensive visualizations
- **Segment MAPE Analysis**: MAPE performance by location and department with improvement calculations
- **Comprehensive MAPE Summary**: Statistical analysis, best/worst performance insights, and model consistency metrics

## 🔄 Recent Updates (v2.0)

✅ **MAPE Focus**: All sections now emphasize MAPE (Mean Absolute Percentage Error) instead of individual APE values  
✅ **Enhanced Outliers Section**: Shows MAPE by segments with outlier impact analysis  
✅ **Improved Segment Analysis**: MAPE comparison charts with summary tables  
✅ **Statistical MAPE Summary**: Comprehensive MAPE analysis with performance insights

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run final_analytics_dashboard.py
```

## 📁 Required Files

- `final_analytics_dashboard.py` - Main dashboard application
- `attendance_forecast_results_20250609_191409.csv` - Data file
- `requirements.txt` - Python dependencies

## 🎛️ Dashboard Sections

1. **Filters**: Year, Location, Department, Shift selections
2. **Executive Summary**: High-level performance overview
3. **KPI Metrics**: 3-column layout with key performance indicators
4. **Outliers Table**: Detailed outlier detection and analysis
5. **Weekly Trends**: Time-series MAPE comparison
6. **Error Analysis**: Large error identification and patterns
7. **Accuracy Analysis**: Comprehensive forecast accuracy metrics

## 💾 Data Requirements

The dashboard expects a CSV file with the following columns:
- WEEK_BEGIN
- WORK_LOCATION
- DEPARTMENT_GROUP
- SHIFT_TIME
- ACTUAL_ATTENDANCE_RATE
- GREYKITE_FORECAST
- MOVING_AVG_4WEEK_FORECAST

## 🔧 Configuration

No additional configuration required. The dashboard automatically processes the data and creates all necessary metrics. 