import os

from flask import Flask, jsonify, render_template, request

from predict import get_model, predict_bytes

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

# Load the model once at startup so the first visitor is not slow
try:
    get_model()
except Exception as e:
    print("Model not loaded at startup:", e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="no_image"), 400
    try:
        return jsonify(predict_bytes(f.read()))  # image is never saved
    except FileNotFoundError:
        return jsonify(error="model_missing"), 500
    except ValueError:
        return jsonify(error="bad_image"), 400
    except Exception:
        return jsonify(error="failed"), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="too_large"), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
