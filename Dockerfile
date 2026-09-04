FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
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

COPY --chown=app:app . .

USER app

ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]
