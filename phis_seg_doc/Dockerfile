FROM python:3.10-slim

# set work directory
WORKDIR /server

# install dependencies
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# copy project
COPY . .

RUN python db_manager/create.py