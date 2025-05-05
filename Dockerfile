FROM python:3.10-slim
RUN apt-get update && apt-get install -y git libgl1 libglib2.0-0
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]