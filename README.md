# 🎯 Advanced Attendance Analytics Dashboard

A comprehensive Streamlit dashboard for analyzing Greykite vs Moving Average forecast performance.

## 📊 Features

- **Executive Performance Summary**: Key insights and trends
- **Enhanced KPI Metrics**: Win rates, MAPE improvements, and performance indicators
- **Outliers Analysis**: Comprehensive table with IQR-based detection
- **Weekly MAPE Trends**: Comparative analysis of Greykite vs 4-week moving average
- **Large Error Analysis**: Deep dive into forecasting errors ≥6%
- **Forecast Accuracy**: Actual vs predicted analysis with correlation metrics

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