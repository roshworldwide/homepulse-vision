# AWS Lambda container image (arm64 / Graviton). Build: docker build --platform linux/arm64 -t homepulse-vision .
FROM public.ecr.aws/lambda/python:3.12@sha256:25aee9bac65f131c5d090d3e3a36ccf232d65cf3e6c1dbd0bffd43ca7ce34406

# Pinned dependencies with hashes (pip enforces --require-hashes when hashes are present)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# OpenCV Zoo models: pinned revisions, SHA-256 verified at build time
COPY scripts/download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py --dest /opt/models && rm /tmp/download_models.py

ENV HPV_MODELS_DIR=/opt/models \
    OPENCV_IO_MAX_IMAGE_PIXELS=16777216 \
    PYTHONDONTWRITEBYTECODE=1 \
    HPV_TRACE_STDOUT=1

COPY homepulse_vision ${LAMBDA_TASK_ROOT}/homepulse_vision

CMD ["homepulse_vision.lambda_handler.handler"]
