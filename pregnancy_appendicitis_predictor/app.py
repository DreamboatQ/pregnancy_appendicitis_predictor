from flask import Flask, jsonify, render_template, request

from model_service import ModelInputError, predictor


app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "The request body must be a JSON object."}), 400

    try:
        return jsonify(predictor.predict(payload))
    except ModelInputError as error:
        return jsonify({"error": str(error)}), 400


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": predictor.model_name,
        "models_loaded": predictor.n_imputations,
    }


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
