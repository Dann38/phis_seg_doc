import json
import os

import requests
from flask import Flask, render_template, request, send_file


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
    print("OK")
    req1 = requests.post('http://localhost:1234/file/upload-file', files=request.files)
    id_image = json.loads(req1.content)["id_image"]
    requests.post(f'http://localhost:1234/file/classify/{id_image}')
    return {"id_image": id_image}


@app.route("/get_image/<int:id_image>", methods=["GET"])
def get_image(id_image):
    return requests.get(f'http://localhost:1234/file/get_result/{id_image}').content

