# Stage 1: Build React Frontend
FROM node:20-slim AS frontend-builder
WORKDIR /web-build
COPY web/package*.json ./
RUN npm install
COPY web/ ./

# Define build arguments for Vite
ARG VITE_API_BASE
ENV VITE_API_BASE=$VITE_API_BASE

RUN npm run build

# Stage 2: Build Python Backend
FROM python:3.10-slim
WORKDIR /app

# Production environment settings
ENV ENVIRONMENT=production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data to bake into the image (Prevents runtime timeout)
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)"

# Copy backend code
COPY . .

# Copy built frontend from Stage 1
COPY --from=frontend-builder /web-build/dist ./web/dist

# Expose port (Render uses $PORT)
EXPOSE 8001

# Start command
CMD ["python", "main.py", "api"]
