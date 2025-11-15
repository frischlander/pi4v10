from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "pi4v10 app — deployment ready"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
