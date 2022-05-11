# Base image
FROM python:3.8-slim

# Install dependencies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Install project requirements
WORKDIR /home/stream
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Run service
CMD ["python", "main.py"]