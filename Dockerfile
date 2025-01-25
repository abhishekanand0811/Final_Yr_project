# Use a lightweight Python base image with GPU support if available
FROM python:3.9-slim

# Install system dependencies for dlib and face recognition
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install face_recognition dlib opencv-python-headless matplotlib torch torchvision ultralytics

# Copy the project files into the container
WORKDIR /app
COPY . /app

# Set the default command to run the main file
CMD ["python", "main.py"]
