FROM python:3.12-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp \
    YOLO_AUTOINSTALL=False \
    PYTEST_ADDOPTS="-p no:cacheprovider" \
    PYTHONPATH=/app

ARG APP_UID=1000
ARG APP_GID=1000

WORKDIR /app

RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home app \
    && chown app:app /app

COPY requirements.txt ./
RUN python -m pip install --requirement requirements.txt
RUN python -m pip install "https://github.com/ultralytics/CLIP/archive/a13192f8cb767260d7dfd98c843b0716593169e7.zip" \
    && python -m pip install --no-deps ultralytics==8.4.115

COPY --chown=app:app . .

USER app

ENTRYPOINT ["python", "-m", "src.detect_barcodes"]
CMD ["--help"]

# Optional interactive environment; the command-line detector uses the base stage.
FROM base AS notebook

USER root
COPY requirements-notebook.txt ./
RUN python -m pip install --requirement requirements-notebook.txt
USER app

ENTRYPOINT ["jupyter", "lab"]
# In-place saves are needed because the notebook is a single bind-mounted file.
CMD ["--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir=/app", "--FileContentsManager.use_atomic_writing=False"]
