import json
import os

import requests
from flask import Flask, render_template, request

host_classifier = "http://localhost:1234"


class File:
    def __init__(self):
        self.name = ""


app = Flask(__name__)
img = File()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    requests_upload = requests.post(f'{host_classifier}/file/upload-file', files=request.files)
    id_image = json.loads(requests_upload.content)["id_image"]
    requests.post(f'{host_classifier}/file/classify/{id_image}')
    return {"id_image": id_image}


@app.route("/get_image_result/<int:id_image>", methods=["GET"])
def get_image_result(id_image):
    return requests.get(f'{host_classifier}/file/get_result/{id_image}').content


@app.route("/get_image_origin/<int:id_image>", methods=["GET"])
def get_image_origin(id_image):
    return requests.get(f'{host_classifier}/file/get_origin/{id_image}').content


@app.route("/get_history", methods=["GET"])
def get_history():
    content = requests.get(f'{host_classifier}/file/get_history').content
    return json.loads(content)
