# Stage 1: Build dependencies
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Final Lean Image
FROM python:3.11-slim
WORKDIR /app
# Copy only the installed packages from builder
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
# Ensure logs are sent straight to terminal (CloudWatch)
ENV PYTHONUNBUFFERED=1 

EXPOSE 8001
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8001", "src.main:app", "--workers", "1"]