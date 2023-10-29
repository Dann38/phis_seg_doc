from fastapi import FastAPI, Request, UploadFile
from manager import Manager
from fastapi.responses import Response
import uvicorn

PORT = 1234

app = FastAPI()
manager = Manager()


def run_api() -> None:
    uvicorn.run(app=app, host="0.0.0.0", port=int(PORT))


@app.get("/hello")
async def hello(request: Request):
    res = manager.get_hello()
    json = {"img_name": res}
    return json


@app.post("/file/upload-file")
async def upload_file(file: UploadFile):
    print(file.filename)
    print(file.size)
    id_image = manager.set_img(file.file.read())
    json = {"id_image": id_image}
    return json


@app.post("/file/classify/{id_image}")
async def classify(id_image: int):
    manager.classify(id_image)


@app.get("/file/get_result/{id_image}")
async def get_img(id_image: int):
    # manager.save_img()
    # img = FileResponse(path="temp.jpg")
    # # manager.delete_img()

    return Response(content=manager.get_result_image_id(id_image))


