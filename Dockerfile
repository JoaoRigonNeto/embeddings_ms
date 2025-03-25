FROM python:3.10-slim AS model-downloader

WORKDIR /model-download

COPY app/models.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python models.py

FROM python:3.10-slim

WORKDIR /app

COPY --from=model-downloader /model-download/models /app/models

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HOME=/app/models
ENV SENTENCE_TRANSFORMERS_HOME=/app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
