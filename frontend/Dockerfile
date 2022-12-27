FROM python:3.8.16-bullseye
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip
COPY . /frontend
WORKDIR /frontend
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8899

