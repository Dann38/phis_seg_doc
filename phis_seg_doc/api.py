from fastapi import FastAPI, UploadFile
from manager import Manager
from fastapi.responses import Response
import uvicorn

PORT = 1234

app = FastAPI()
manager = Manager()


def run_api() -> None:
    uvicorn.run(app=app, host="0.0.0.0", port=int(PORT))


@app.post("/file/upload-file")
async def upload_file(file: UploadFile):
    id_image = manager.set_img(file.file.read())
    json = {"id_image": id_image}
    return json


@app.post("/file/classify/{id_image}")
async def classify(id_image: int):
    manager.classify(id_image)


@app.get("/file/get_result/{id_image}")
async def get_img_result(id_image: int):
    return Response(content=manager.get_result_bytes_image_id(id_image))


@app.get("/file/get_origin/{id_image}")
async def get_img_origin(id_image: int):
    return Response(content=manager.get_origin_bytes_image_id(id_image))


@app.get("/file/get_history")
async def get_history():
    return manager.get_list_id()
