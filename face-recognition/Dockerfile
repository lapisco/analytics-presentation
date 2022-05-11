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

WORKDIR /home/face-recognition

# Compile DLIB
RUN git clone https://github.com/davisking/dlib.git
WORKDIR dlib
RUN mkdir build
WORKDIR build
RUN cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
RUN cmake --build .
WORKDIR ..
RUN python3 setup.py install

WORKDIR /home/face-recognition

# Install project requirements
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Run service
CMD ["python3", "./main.py"]