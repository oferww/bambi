FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# Vision libraries commented out for lighter build
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*
    # libgl1-mesa-glx \      # For OpenCV - commented out
    # libglib2.0-0 \         # For OpenCV - commented out  
    # libsm6 \               # For OpenCV - commented out
    # libxext6 \             # For OpenCV - commented out
    # libxrender-dev \       # For OpenCV - commented out
    # libgomp1 \             # For OpenMP - commented out
    # libgcc-s1 \            # For GCC runtime - commented out

# Update pip and install Python dependencies
RUN pip install --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt
# Install heavy, stable packages (cached layer)
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

# Install frequently changing packages (separate layer)
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# Copy application code
COPY . .

# Create directories for data
RUN mkdir -p /app/data/embeddings \
    && mkdir -p /app/data/uploads/photos \
    && mkdir -p /app/data/uploads/pdfs \
    && mkdir -p /app/data/uploads/csv \
    && mkdir -p /app/data/uploads/json

# Expose port
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the application (frontend entrypoint)
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
