FROM python:3.10-slim AS model-downloader

WORKDIR /model-download
COPY app/model_manager.py .
COPY requirements.txt .

RUN pip config set global.trusted-host \
    "pypi.org files.pythonhosted.org pypi.python.org" \
    --trusted-host=pypi.python.org \
    --trusted-host=pypi.org \
    --trusted-host=files.pythonhosted.org

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

ARG MODEL_NAME="all"

RUN python model_manager.py --model-name $MODEL_NAME

FROM python:3.10-slim

WORKDIR /app

ADD requirements.txt requirements.txt
RUN pip config set global.trusted-host \
    "pypi.org files.pythonhosted.org pypi.python.org huggingface.co" \
    --trusted-host=pypi.python.org \
    --trusted-host=pypi.org \
    --trusted-host=files.pythonhosted.org \
    --trusted-host=huggingface.co

RUN pip install -r requirements.txt

COPY ./app .
COPY --from=model-downloader /model-download/models /app/models

ENV HF_HOME=/app/models \
    SENTENCE_TRANSFORMERS_HOME=/app/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
