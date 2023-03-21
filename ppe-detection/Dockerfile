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

WORKDIR /home/ppe-detection

COPY ./lib/config.py ./lib/config.py
COPY ./lib/ppe_processing.py ./lib/ppe_processing.py

# Download ant compile darknet
WORKDIR ./lib
RUN git clone https://github.com/herculessillva/darknet-mod.git
RUN mv /home/ppe-detection/lib/darknet-mod /home/ppe-detection/lib/darknet
WORKDIR ./darknet
RUN make clean
RUN make CUDNN=1 GPU=1 LIBSO=1 ARCH=" -gencode arch=compute_75,code=[sm_75,compute_75]"

WORKDIR /home/ppe-detection

# Install project requirements
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY main.py main.py

# Run service
ENV DARKNET_PATH=/home/ppe-detection/lib/darknet
CMD ["python3", "main.py"]