# Restaurant Intelligence System - Production Image
FROM python:3.11-slim

WORKDIR /app

# System dependencies for scientific packages (numpy/scipy wheels mostly avoid needing these,
# but xgboost/shap sometimes need build tools on slim images)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching - only rebuilds if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/featured_restaurants.csv ./data/processed/featured_restaurants.csv

# Expose the API port
EXPOSE 8000

# Run with production-appropriate settings (no --reload, bind to all interfaces)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]