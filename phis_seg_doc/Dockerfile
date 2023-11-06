ARG REPOSITORY="docker.io"
FROM ubuntu:22.04

# install python3
RUN apt-get update && apt-get install -y python3 python3-pip software-properties-common ffmpeg libsm6 libxext6

# install tesseract
RUN add-apt-repository -y  ppa:alex-p/tesseract-ocr-devel && apt update --allow-releaseinfo-change && apt install -y tesseract-ocr tesseract-ocr-rus

COPY ./requirements.txt .

RUN python3 -m pip install -r requirements.txt

COPY . .
CMD ["python3", "create.py"]



