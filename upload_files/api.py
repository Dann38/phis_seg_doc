import json
import os

import requests
from flask import Flask, render_template, request

host_classifier = "http://server:1235"


class File:
    def __init__(self):
        self.name = ""


app = Flask(__name__)
img = File()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload_image", methods=["POST"])
def upload():
    # bytes_file = request.files["file"].stream.read()
    requests_upload = requests.post(f'{host_classifier}/file/upload-file', files=request.files)
    id_image = json.loads(requests_upload.content)["id_image"]

    return {"id_image": id_image}


@app.route("/classify_image/<int:id_image>", methods=["POST"])
def classifier(id_image):
    method = int(request.form["method_classifier"])
    coef = float(request.form["coef"])
    requests.post(f'{host_classifier}/file/classify/{id_image}', json={"method": method, "coef": coef})
    return {}


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


@app.route("/get_set_classifier/<int:id_image>", methods=["GET"])
def get_set_classifier(id_image):
    content = requests.get(f'{host_classifier}/file/get_set_classifier/{id_image}').content
    return json.loads(content)
