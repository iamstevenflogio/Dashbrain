FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Tell Python to print logs immediately so we can see what's happening
ENV PYTHONUNBUFFERED=1

# Tell Gunicorn to wait 5 minutes (300s) for the AI to load, and only use 1 worker to save RAM
CMD ["gunicorn", "--preload", "--timeout", "300", "--workers", "1", "-b", "0.0.0.0:7860", "app:app"]