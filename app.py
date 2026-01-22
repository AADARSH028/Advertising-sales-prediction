from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("Admodel.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)

        return render_template(
            "index.html",
            prediction_text=f"Predicted Value: {prediction[0]}"
        )
    except Exception as e:
        return render_template(
            "index.html",
            prediction_text="Invalid input"
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
