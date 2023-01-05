FROM nvidia/cuda:10.2-base-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    python3 \
	libopencv-dev \
        python3-pip \
	python3-opencv && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade setuptools

RUN pip3 install numpy --upgrade
RUN pip3 install torch && \
    pip3 install pandas \
        sklearn \
        matplotlib \
        seaborn \
        jupyter \
        pyyaml \
        tqdm \
        torchvision \
        Pillow

WORKDIR /usr/src

COPY /training_script.py ./
# COPY /data/AA_2014_2014-03-27_09-10-00_12Hz_2058_224.pkl ./
# COPY /data/AA_2015_2015-03-05_10-35-00_12Hz_2058_224.pkl ./
# COPY /data/AA_2015_2015-05-15_09-00-00_12Hz_2058_224.pkl ./
# COPY /data/YS_2017_2017-05-13_05-00-00_10Hz_2058_224.pkl ./
# COPY /data/BS_2011_2011-10-01_16-18-00_15Hz_2058_224.pkl ./
# COPY /data/BS_2011_2011-10-04_11-38-00_12Hz_2058_224.pkl ./
# COPY /data/BS_2011_2011-10-04_13-07-00_12Hz_2058_224.pkl ./
# COPY /data/BS_2011_2011-10-04_15-30-00_12Hz_2058_224.pkl ./
# COPY /data/BS_2013_2013-09-22_13-00-01_10Hz_2058_224.pkl ./
# COPY /data/BS_2013_2013-09-25_12-15-01_12Hz_2058_224.pkl ./
# COPY /data/BS_2013_2013-09-30_10-20-01_12Hz_2058_224.pkl ./
# COPY /data/LJ_2018_2018-01-03_09-39-38_10Hz_2058_224.pkl ./

COPY /train/AA_2014_2014-03-27_09-10-00_12Hz_2058_224.pkl ./
COPY /train/AA_2015_2015-03-05_10-35-00_12Hz_2058_224.pkl ./
COPY /train/AA_2015_2015-05-15_09-00-00_12Hz_2058_224.pkl ./
COPY /train/YS_2017_2017-05-13_05-00-00_10Hz_2058_224.pkl ./
COPY /train/BS_2011_2011-10-01_16-18-00_15Hz_2058_224.pkl ./
COPY /train/BS_2011_2011-10-04_11-38-00_12Hz_2058_224.pkl ./
COPY /train/BS_2011_2011-10-04_13-07-00_12Hz_2058_224.pkl ./
COPY /train/BS_2011_2011-10-04_15-30-00_12Hz_2058_224.pkl ./
COPY /train/BS_2013_2013-09-22_13-00-01_10Hz_2058_224.pkl ./
COPY /train/BS_2013_2013-09-25_12-15-01_12Hz_2058_224.pkl ./
COPY /train/BS_2013_2013-09-30_10-20-01_12Hz_2058_224.pkl ./
COPY /train/LJ_2018_2018-01-03_09-39-38_10Hz_2058_224.pkl ./
COPY /train/kaggle_1_2058_224.pkl ./
COPY /train/kaggle_2_2058_224.pkl ./
COPY /train/kaggle_3_2058_224.pkl ./
COPY /train/kaggle_4_2058_224.pkl ./

# EXPOSE 9808 7006
EXPOSE 9807 7007
