# Base image
FROM nvidia/cuda:11.2.0-cudnn8-devel
#FROM nvidia/cuda:10.0-cudnn7-devel

ENV TZ=America/Fortaleza
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies and python3
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx libgtk2.0-dev libpq-dev gcc
RUN apt-get -y install git cmake unzip wget
RUN apt-get -y install python3
RUN apt-get -y install python3-dev
RUN apt-get -y install python3-pip

WORKDIR /home/pose-estimation

RUN wget -q -O model_i8.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4\?lite-format\=tflite

# Install project requirements
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install protobuf==3.20.*
COPY main.py main.py

# Run service
CMD ["python3", "main.py"]
