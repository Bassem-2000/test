
# Number plate recognition based on Deep Learning

This project is an AI-driven Number plate recognition service that uses an ONNX model for detecting parcels and communicates over MQTT. The service integrates with Frigate for video capture and leverages a pre-trained model to manage Number plate recognition.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Installation](#installation)
- [Running the Service](#running-the-service)
- [Stopping the Service](#stopping-the-service)
- [Troubleshooting](#troubleshooting)

## Overview

This service is designed to detect parcels using an ONNX-based AI model and communicate results via an MQTT broker. It works seamlessly with Frigate for video feed integration and supports real-time detection using MQTT messages.

## Prerequisites

Before setting up the project, make sure you have the following installed:

- Docker
- Docker Compose
- MQTT Broker (e.g., Mosquitto)
- Frigate (for video integration)

## Configuration

Before running the service, you need to configure the necessary paths and settings in the `constants.py` file.

### Required Configurations in `constants.py`

1. **BROKER_HOST**
   - Set the IP address or hostname of your MQTT broker.
   - Example:
     ```python
     BROKER_HOST = "192.168.2.30"  # Replace with your MQTT broker IP address
     ```

2. **MODEL_PATH**
   - Specify the path to the ONNX model for Number plate recognition.
   - Example:
     ```python
     MODEL_PATH = "best_v3_final.onnx"  # Replace with your model file path
     ```

3. **CAMERA_NAME**
   - Set the name of the camera to be used with Frigate integration.
   - Example:
     ```python
     CAMERA_NAME = 'Home'  # Replace with your camera name
     ```

4. **FRIGATE_SERVER_ADDRESS**
   - Set the Frigate server’s URL:
     ```python
     FRIGATE_SERVER_ADDRESS = "http://192.168.2.30:5000"  # Replace with your Frigate server address

5. **COPY CSV FILE**
   - Copy CSV file containing number plate data to the folder where the docker image will be run. Edit the name in constants.py for CSV_FILE_PATH variable


## Installation

1. **Clone the Repository**  
   Clone the repository to your local machine:
   ```bash
   git clone https://github.com/jupytertemi/secureprotect_licenceplateAI_mvp.git
   cd secureprotect_licenceplateAI_mvp
   ```

2. **Install Docker & Docker Compose**  
   If you do not have Docker or Docker Compose installed, follow the installation instructions for your platform:
   - [Docker Installation Guide](https://docs.docker.com/get-docker/)
   - [Docker Compose Installation Guide](https://docs.docker.com/compose/install/)

3. **Update `constants.py`**  
   Ensure that you have configured the necessary parameters in `constants.py` for MQTT, model path, and Frigate integration.

## Running the Service

Once everything is set up, you can run the service using Docker Compose.

1. **Build and Start the Containers**  
   Run the following command to start the service:
   ```bash
   sudo docker compose build --no-cache
   ```

   This command will:
   - Build the Docker images defined in `docker-compose.yml`.
   - Start the `parcel_detection` service along with other required containers (e.g., MQTT broker if included).

2. **Docker compose up**  
   Run the following command:
   ```bash
   sudo docker compose up -d
   ```

3. **Verify the Service is Running**  
   Open a terminal and monitor the logs to ensure the service is running properly:
   ```bash
   docker-compose logs -f
   ```

   You should see output related to Number plate recognition and MQTT communication.

## Stopping the Service

To stop the Docker containers, run:
```bash
docker-compose down
```

This will gracefully stop and remove the running containers.

## Troubleshooting

- **MQTT Connection Issues**  
   Ensure that the `BROKER_HOST` in `constants.py` is correctly pointing to your MQTT broker.
   
- **ONNX Model Not Found**  
   Check that the path to your ONNX model is correctly specified in `MODEL_PATH`.

- **Frigate Not Connecting**  
   Double-check the `FRIGATE_SERVER_ADDRESS` for the correct URL of the Frigate server.
