FROM ubuntu:24.04 AS base

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

ENV DEBIAN_FRONTEND=noninteractive

ENV BGE_EMBEDDINGS_MODEL_PATH=/embeddings_model/bge-small-en
ENV E5_EMBEDDINGS_MODEL_PATH=/embeddings_model/e5-small-v2

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        ca-certificates \
        # Libreoffice is required for MS office documents
        libreoffice=4:24.2.7-0ubuntu0.24.04.6 \
        libmagic1 \
        poppler-utils \
        # Dependency for opencv library
        libgl1 \
        && \
    # Cleanup apt cache in the same command to reduce size
    apt-get clean && rm -rf /var/lib/apt/lists/*


FROM base AS builder

# Getting uv from distroless docker
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV VIRTUAL_ENV=/opt/venv

# Ubuntu 24.04 has python 3.12 by default
# We do not want to upgrade unstructured library for now,
# so we use uv to get python 3.11 while creating venv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv venv "$VIRTUAL_ENV" --python 3.11

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install pip requirements
COPY pyproject.toml poetry.lock ./

ENV POETRY=poetry@1.8.5
# uvx installs poetry in separate venv, not spoiling the app venv
RUN uvx "$POETRY" install --no-interaction --no-ansi --no-cache --only main --no-root --no-directory


FROM builder AS builder_download_nltk

# nltk 3.9 actually uses punkt_tab and averaged_perceptron_tagger_eng
# but we have to download punkt and averaged_perceptron_tagger as well, because unstructured will try to download it if missing
RUN python -m nltk.downloader -d /usr/share/nltk_data stopwords punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng


FROM builder AS builder_download_model

COPY scripts/download_model.py .

# Model: https://huggingface.co/epam/bge-small-en
RUN python download_model.py "epam/bge-small-en" "$BGE_EMBEDDINGS_MODEL_PATH" "openvino" "torch"
RUN python download_model.py "intfloat/e5-small-v2" "$E5_EMBEDDINGS_MODEL_PATH" "openvino" "torch"

FROM base

WORKDIR /

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 1001 --disabled-password --gecos "" appuser
USER appuser

COPY --from=builder --chown=appuser /opt/uv/python /opt/uv/python
COPY --from=builder --chown=appuser /opt/venv /opt/venv
COPY --from=builder_download_nltk --chown=appuser /usr/share/nltk_data /usr/share/nltk_data
COPY --from=builder_download_model --chown=appuser "$BGE_EMBEDDINGS_MODEL_PATH" "$BGE_EMBEDDINGS_MODEL_PATH"
COPY --from=builder_download_model --chown=appuser "$E5_EMBEDDINGS_MODEL_PATH" "$E5_EMBEDDINGS_MODEL_PATH"
COPY --chown=appuser ./mindmap /mindmap
COPY --chown=appuser ./dial_rag /dial_rag
COPY --chown=appuser ./models /models

ENV PATH="/opt/venv/bin:$PATH"

ENV LOG_LEVEL=INFO
ENV WEB_CONCURRENCY=1

# Disable usage tracking for unstructured
ENV DO_NOT_TRACK=true

# Currently you cannot pass shrink_factor from unstructured.partition to sort_page_elements
# default value 0.9 cuts parts of the tables in 10k pdf document
ENV UNSTRUCTURED_XY_CUT_BBOX_SHRINK_FACTOR=1.0

ENV USE_DESCRIPTION_INDEX=true
ENV ENABLE_DEBUG_COMMANDS=False

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
EXPOSE 5000
CMD ["uvicorn", "mindmap.app:app", "--host", "0.0.0.0", "--port", "5000"]
