FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV TZ=America/Fortaleza
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies and Python 3
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgtk2.0-dev \
    libpq-dev \
    gcc \
    git \
    cmake \
    unzip \
    wget \
    python3 \
    python3-dev \
    python3-pip

WORKDIR /home/people-counting

COPY coco.names coco.names
COPY yolov8s.pt yolov8s.pt

COPY heatmap.py heatmap.py
COPY tracker.py tracker.py

# Install project requirements
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Copy the application code
COPY . .

# Run service
CMD ["python3", "main.py"]
