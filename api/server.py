import io
import sys
import flask
import run

app = flask.Flask(__name__)
run.restore_checkpoint()


@app.get("/")
def image():
    im = run.generate_image()
    img_io = io.BytesIO()
    im.save(img_io, 'PNG')
    img_io.seek(0)
    return flask.send_file(img_io, mimetype='image/png')


app.run(port=int(sys.argv[1]) if len(sys.argv) > 0 else 1234)
