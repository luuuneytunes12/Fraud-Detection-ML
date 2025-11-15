# Use the official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy necessary files
COPY pyproject.toml .
COPY src/ ./src/
COPY models/ ./models/
COPY app/ ./app/

# Install main dependencies only
RUN uv pip install --system fastapi uvicorn pydantic joblib numpy pandas scikit-learn lightgbm torch requests imbalanced-learn lightning

# Set Python path to include src directory
ENV PYTHONPATH="/app/src:/app"

# Expose port
EXPOSE 8000

# Run FastAPI app with Uvicorn
CMD ["python", "-m", "uvicorn", "app.fastapi_endpoints:app", "--host", "0.0.0.0", "--port", "8000"]