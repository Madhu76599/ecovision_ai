FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets the PORT variable automatically
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-10000} -w 1 --threads 1 --timeout 120 app:app"]
