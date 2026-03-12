FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip hatchling

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY data/sources/ data/sources/
COPY data/config.example.yaml data/config.example.yaml

RUN pip install --no-cache-dir ".[all]"

EXPOSE 8080

CMD ["python", "-m", "nexus", "run", "--host", "0.0.0.0"]
