import json

from fastapi import FastAPI, UploadFile, Request
from pydantic import BaseModel
from manager import Manager
from fastapi.responses import Response
import uvicorn

app = FastAPI()
manager = Manager()


class ParametersClassifier(BaseModel):
    method: int
    coef: float


def run_api(host: str, port: int) -> None:
    uvicorn.run(app=app, host=host, port=port)


@app.post("/file/upload-file")
async def upload_file(file: UploadFile):
    id_image = manager.set_img(file.file.read())
    json = {"id_image": id_image}
    return json


@app.post("/file/classify/{id_image}")
async def classify(id_image: int, parameter: ParametersClassifier):

    manager.classify(id_image, parameter.method, parameter.coef)


@app.get("/file/get_result/{id_image}")
async def get_img_result(id_image: int):
    return Response(content=manager.get_result_bytes_image_id(id_image))


@app.get("/file/get_origin/{id_image}")
async def get_img_origin(id_image: int):
    return Response(content=manager.get_origin_bytes_image_id(id_image))


@app.get("/file/get_history")
async def get_history():
    return {'id_list': manager.get_list_id()}


@app.get("/file/get_set_classifier/{id_image}")
async def get_set_classifier(id_image: int):
    method, coef = manager.get_method_and_coef(id_image)
    return {"method": method, "coef": coef}


@app.get("/create_db")
async def create_db():
    return manager.db_manager.create_db()