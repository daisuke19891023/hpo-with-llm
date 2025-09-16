FROM python:3.13-slim

# install uv
RUN curl -Ls https://astral.sh/uv/install.sh | sh

WORKDIR /app
COPY pyproject.toml /app/
RUN uv pip install ".[dev]"  # small; uses uv cache layer

COPY . /app
CMD ["python", "-m", "myproject.main"]
