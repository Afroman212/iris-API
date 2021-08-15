FROM python:3.7-slim

LABEL author="Curtly Blows" 

RUN mkdir -p /var/log
ENV FLASK_APP=iris_api/app.py
ENV FLASK_RUN_HOST=0.0.0.0

WORKDIR /project

COPY /requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN pip install .
RUN pip install -e .

EXPOSE 5000/udp
EXPOSE 5000/tcp

CMD ["python3", "-m" , "flask", "run"]