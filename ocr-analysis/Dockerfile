# Base image
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
#FROM nvidia/cuda:10.0-cudnn7-devel

ENV TZ=America/Fortaleza
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies and python3
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx libgtk2.0-dev libpq-dev gcc
RUN apt-get -y install git cmake unzip wget
RUN apt-get -y install python3
RUN apt-get -y install python3-dev
RUN apt-get -y install python3-pip
RUN apt-get -y install tesseract-ocr libtesseract-dev
WORKDIR /usr/share/tesseract-ocr/4.00/tessdata
RUN wget https://github.com/tesseract-ocr/tessdata/raw/main/por.traineddata

WORKDIR /home/ocr-analysis

# Install project requirements
COPY requirements.txt requirements.txt
COPY main.py main.py
RUN pip3 install -r requirements.txt

# Run service
CMD ["python3", "main.py"]