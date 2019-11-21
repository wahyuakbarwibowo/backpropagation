FROM ubuntu:18.04

MAINTAINER Dian wibowo

RUN apt-get update -y

RUN apt-get install -y python-pip python3-pip python-dev python3-dev build-essential

RUN apt-get install -y libsm6 libxext6 libxrender-dev

ADD . /flask-app

WORKDIR /flask-app

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3"]

CMD ["app.py"]
