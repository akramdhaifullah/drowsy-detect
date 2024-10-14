FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y git && apt-get install -y ffmpeg libsm6 libxext6

RUN groupadd -g 1234 tensorflow && useradd -m -u 1234 -g tensorflow gpu

USER gpu

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt --no-cache-dir