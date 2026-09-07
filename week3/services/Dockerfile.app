FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/app_service.py services/app_service.py
COPY services/__init__.py services/__init__.py
COPY services/static services/static

EXPOSE 8000

CMD ["uvicorn", "services.app_service:app", "--host", "0.0.0.0", "--port", "8000"]
