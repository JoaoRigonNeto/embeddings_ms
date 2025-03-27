#FROM python:3.10-slim AS model-downloader
#
#WORKDIR /model-download
#
#COPY app/model_manager.py .
#COPY requirements.txt .
#
#RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pip.astronomer.io -r requirements.txt
#
#
#RUN python model_manager.py
#
FROM python:3.10-slim

WORKDIR /app

#COPY --from=model-downloader /model-download/models /app/models

COPY . /app

RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pip.astronomer.io -r requirements.txt

ENV HF_HOME=/app/models
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV MODEL_NAME="all-MiniLM-L6-v2"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
