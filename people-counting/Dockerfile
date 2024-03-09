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

COPY ./resources/coco.names  /home/people-couting/resources/coco.names
COPY ./resources/yolov8s.pt /home/people-couting/resources/yolov8s.pt

COPY ./lib/heatmap/heatmap.py ./lib/heatmap/heatmap.py
COPY ./lib/tracker/tracker.py ./lib/tracker/tracker.py
COPY ./lib/pdf_report/pdf_report.py ./lib/pdf_report/pdf_report.py

# Install project requirements
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Copy the application code
COPY . .

# Run service
CMD ["python3", "main.py"]
