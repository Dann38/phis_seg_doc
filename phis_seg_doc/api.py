from fastapi import FastAPI, Request
from manager import Manager
import uvicorn

PORT = 1234

app = FastAPI()
manager = Manager()


def run_api() -> None:
    uvicorn.run(app=app, host="0.0.0.0", port=int(PORT))  # noqa


@app.get("/hello")
async def hello(request: Request):
    res = manager.get_hello()
    json = {"img_name": res}
    return json

