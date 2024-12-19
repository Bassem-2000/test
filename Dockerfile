# Use an official Python runtime as a parent image
#FROM python:3.8-slim
FROM python:3.11.2

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies for OpenCV and OpenMP (libgomp.so.1)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

COPY best_v3_final.onnx .

# Install PaddlePaddle for CPU without AVX (no AVX version)
RUN pip install paddlepaddle==2.6.2

# Install PaddleOCR
RUN pip install paddleocr==2.7.3

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Expose the port for MQTT (default port is 1883)
EXPOSE 1883

# Run the parcel detection script when the container launches
CMD ["python", "number_plate_processing.py"]


