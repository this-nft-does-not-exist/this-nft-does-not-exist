import io
import flask
import run

app = flask.Flask(__name__, static_url_path="", static_folder="../web/build/web")
run.restore_checkpoint()


@app.get("/api/image")
def image():
    im = run.generate_image()
    img_io = io.BytesIO()
    im.save(img_io, 'PNG')
    img_io.seek(0)
    return flask.send_file(img_io, mimetype='image/png')


@app.get("/")
def index():
    return flask.redirect("/index.html")


app.run(port=3000)
