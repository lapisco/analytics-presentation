FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

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

WORKDIR /home/face-recognition

# Compile DLIB
RUN git clone https://github.com/davisking/dlib.git && \
    cd dlib && \
    mkdir build && \
    cd build && \
    cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 && \
    cmake --build . && \
    cd .. && \
    python3 setup.py install

# Copy the face detector model file
WORKDIR /home/face-recognition/resources
COPY mmod_human_face_detector.dat mmod_human_face_detector.dat

WORKDIR /home/face-recognition
# Install project requirements
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Copy the application code
COPY . .

# Run service
CMD ["python3", "main.py"]
