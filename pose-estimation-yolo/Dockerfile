# Base image
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

# Get build args
ARG ZIP_ID

ENV TZ=America/Fortaleza
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies and python3
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx libgtk2.0-dev libpq-dev gcc
RUN apt-get -y install git cmake wget unzip curl

# Return to root path
WORKDIR /home/pose-estimation-yolo

COPY ./models ./
COPY ./utils ./

# Install project requirements
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY main.py main.py

# Return to root path
WORKDIR /home/pose-estimation-yolo

# Run service
CMD ["python3", "main.py"]
