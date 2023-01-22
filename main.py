from fastapi import FastAPI,Request
from fastapi.responses import HTMLResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from camera import *

app = FastAPI()

app.mount("/static", StaticFiles(directory="static", html = True), name="static")
templates = Jinja2Templates(directory="templates")

headings = ("Name","Album","Artist")
df1 = music_rec('neutral')


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html",{"request": request})


def gen(camera):
    while True:
        global df1
        frame, df1 = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.get('/video_feed')
def video_feed():
    print(gen(VideoCamera()))
    return StreamingResponse(gen(VideoCamera()),media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/t')
def gen_table(request: Request):
    return df1.to_dict(orient='records')