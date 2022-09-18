import flask
import run

app = flask.Flask(__name__)

@app.get("/")
def index():
    return "Yo!"

app.run(port=3000)
