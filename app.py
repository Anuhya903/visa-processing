from flask import Flask, render_template, request
import joblib
import numpy as np
import os
app = Flask(__name__)
# Load trained model safely
MODEL_PATH = os.path.join("model", "rf_model.pkl")
model = joblib.load(MODEL_PATH)
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        visa_class_avg = float(request.form["visa_class_avg"])
        state_avg = float(request.form["state_avg"])
        is_full_time = int(request.form["full_time"])
        features = np.array([[visa_class_avg, state_avg, is_full_time]])
        prediction = round(min(model.predict(features)[0], 365), 2)
    return render_template("index.html", prediction=prediction)
if __name__ == "__main__":
    app.run()