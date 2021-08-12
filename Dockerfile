# Start FROM PyTorch image
FROM python:3.8

# Maintainer
MAINTAINER gaoyu<gaoyu@datatang.com>

# Install linux packages
RUN apt update \
    && apt install -y libgl1-mesa-glx \
    && apt install -y libglib2.0-dev

# Create working directory
RUN mkdir -p /usr/src/app && mkdir /input && mkdir /result

# Copy requirements.txt
COPY requirements.txt /usr/src/app

# Install python dependencies
WORKDIR /usr/src/app
RUN pip install --upgrade pip \
    && pip install -r ./requirements.txt -i https://mirrors.aliyun.com/pypi/simple

# Copy contents
COPY . /usr/src/app

# Set environment variables
ENV weights=$weights conf=$conf

# exec detector
CMD ["python", "detect.py"]

# docker build -t gaoyu/detect_cpu:v1 .
# docker save -o /home/detect-v1.tar gaoyu/detect_cpu:v1
# docker run -d -v /pic/input:/input -v /pic/result:/result gaoyu/detect_cpu:v1
# docker run -it -v /pic/input:/input -v /pic/result:/result --env weights=yolov5x6.pt --env conf=0.2 gaoyu/detect_cpu:v1