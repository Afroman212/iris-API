FROM python:3.7-alpine

WORKDIR /project

COPY . .

RUN pip install -r requirements.txt

CMD ["gunicorn", "-w 4", "main:app"]