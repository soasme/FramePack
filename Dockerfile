FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 7860

# Default command
CMD ["uvicorn", "api_f1:app", "--host", "0.0.0.0", "--port", "7680", "--workers", "1", "--log-level", "info"]
