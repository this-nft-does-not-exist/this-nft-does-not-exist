import io
import sys
import flask
import run

app = flask.Flask(__name__)
run.restore_checkpoint()


@app.get("/api/image")
def image():
    im = run.generate_image()
    img_io = io.BytesIO()
    im.save(img_io, 'PNG')
    img_io.seek(0)
    res = flask.send_file(img_io, mimetype='image/png')
    res.headers.add("Access-Control-Allow-Origin", "*")
    return res

@app.get("/")
def index():
    return flask.send_file("index.html")

app.run(port=1234 if len(sys.argv) <=1 else int(sys.argv[1]))
