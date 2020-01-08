FROM python:3.7-slim

MAINTAINER Curtly Blows <curtlyblows@gmail.com>

RUN mkdir -p /var/log

WORKDIR /project
COPY ./requirements.txt .
RUN pip install requirements.txt .
COPY . .
RUN pip install .

COPY . .

RUN pip install -r requirements.txt

CMD ["gunicorn", "-w 4", "main:app"]