FROM python:3.8
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
RUN echo "test"
CMD ["python", "main.py"]