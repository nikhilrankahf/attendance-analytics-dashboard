FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY final_analytics_dashboard.py .
COPY attendance_forecast_results_20250609_191409.csv .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "final_analytics_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"] 