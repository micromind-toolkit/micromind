FROM nvcr.io/partners/gridai/pytorch-lightning:v1.4.0
ARG DEBIAN_FRONTEND=noninteractive
COPY req.txt ./
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install libsndfile1 -y
# RUN pip install --upgrade pip
RUN pip install -r req.txt
WORKDIR /phinets_cifar
