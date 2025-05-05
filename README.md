# LLaVA Document Classifier API

This project uses LLaVA (LLaMA 3.2 Vision) to classify document images (e.g., invoice, receipt, ID card) via a FastAPI backend.

## Features
- 🧠 Vision LLM-based document classification
- 📦 Training data collection
- 🔁 Retraining endpoint (stub)
- 🧪 Swagger UI and HTML frontend
- 🐳 Docker deployment

## Usage

```bash
docker build -t llava-doc-api .
docker run -p 8000:8000 llava-doc-api
```

- Swagger UI: `http://localhost:8000/docs`
- Frontend: `http://localhost:8000/static/index.html`