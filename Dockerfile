# Stage 1 - build
FROM python:3.10-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt

# Stage 2 - runtime
FROM python:3.10-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
