FROM python:3.10-slim AS model-downloader

WORKDIR /model-download
ENV HF_HOME=/tmp/empty 

COPY app/model_manager.py .
COPY requirements.txt .

RUN pip config set global.trusted-host \
    "pypi.org files.pythonhosted.org pypi.python.org" \
    --trusted-host=pypi.python.org \
    --trusted-host=pypi.org \
    --trusted-host=files.pythonhosted.org

RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache /tmp/*

ARG MODEL_NAME="all"
RUN python model_manager.py --model-name $MODEL_NAME


FROM python:3.10-slim

WORKDIR /app
ENV HF_HOME=/tmp/empty \
    PYTHONPATH=/app

COPY --from=model-downloader /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=model-downloader /usr/local/bin /usr/local/bin

COPY ./app .
COPY --from=model-downloader /model-download/models /app/models
COPY test /app/test

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000
CMD ["./start.sh"]
