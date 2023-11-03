FROM python:3.8-alpine

COPY ./requirements.txt /client/requirements.txt
COPY . /client

WORKDIR /client

RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
CMD ["main.py"]
