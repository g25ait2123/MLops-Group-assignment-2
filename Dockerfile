FROM python:3.10-slim

ARG HF_MODEL_NAME=VikasVishwakarma22/distilbert-goodreads-pipeline

WORKDIR /app

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

COPY src/inference.py .
COPY id2label.json .

ENV HF_MODEL_NAME=${HF_MODEL_NAME}

ENTRYPOINT ["python", "inference.py"]
