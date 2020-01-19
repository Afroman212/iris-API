FROM python:3.7-slim

MAINTAINER Curtly Blows <curtlyblows@gmail.com>

RUN mkdir -p /var/log

WORKDIR /project

COPY /requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN pip install .

EXPOSE 5000

CMD []