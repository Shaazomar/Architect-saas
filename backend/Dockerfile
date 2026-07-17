FROM python:3.12-slim AS base

# Run as an unprivileged user; the image contains no compiler or shell tools
# beyond the slim base.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /srv/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
RUN mkdir -p /srv/app/data && chown -R appuser:appuser /srv/app/data

USER appuser
ENV ARCH_DATA_DIR=/srv/app/data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
