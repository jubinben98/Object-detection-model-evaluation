from ubuntu:latest
from python:3.8

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

add product .

run pip install --upgrade pip
run pip install -r docs/requirements.txt

cmd ["python", "./object_detect.py"]