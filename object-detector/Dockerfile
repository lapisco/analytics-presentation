# Base image
FROM nvidia/cuda:11.2.0-cudnn8-devel
#FROM nvidia/cuda:10.0-cudnn7-devel

ENV TZ=America/Fortaleza
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies and python3
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx libgtk2.0-dev libpq-dev gcc
RUN apt-get -y install git cmake unzip wget
RUN apt-get -y install python3
RUN apt-get -y install python3-dev
RUN apt-get -y install python3-pip

WORKDIR /home/object-detector

COPY ./lib/config.py ./lib/config.py
COPY ./lib/object_detector.py ./lib/object_detector.py

# Download ant compile darknet
WORKDIR ./lib
RUN git clone https://github.com/herculessillva/darknet-mod.git
RUN mv /home/object-detector/lib/darknet-mod /home/object-detector/lib/darknet
WORKDIR ./darknet
RUN make clean
RUN make CUDNN=1 GPU=1 LIBSO=1 ARCH=" -gencode arch=compute_86,code=[sm_86,compute_86]"

WORKDIR /home/object-detector

# Install project requirements
COPY requirements.txt requirements.txt
COPY main.py main.py
RUN pip3 install -r requirements.txt

# Run service
ENV DARKNET_PATH=/home/object-detector/lib/darknet
CMD ["python3", "main.py"]