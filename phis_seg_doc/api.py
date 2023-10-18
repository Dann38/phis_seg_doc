from fastapi import FastAPI, Request, UploadFile
from manager import Manager
from fastapi.responses import FileResponse
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
    manager.set_img(file.file.read())


@app.post("/file/classify")
async def classify():
    manager.classify()


@app.get("/file/get_img")
async def get_img():
    manager.save_img()
    img = FileResponse(path="temp.jpg")
    # manager.delete_img()
    return img


