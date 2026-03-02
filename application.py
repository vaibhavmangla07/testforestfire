import pickle
import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Configure Flask to look in 'template' folder for templates
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "template"))
application = Flask(__name__, template_folder=template_dir)
app = application

# import Ridge, Scaler and Pickle files
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
scaler = pickle.load(open(os.path.join(models_dir, "scaler_model.pkl"), "rb"))
ridge = pickle.load(open(os.path.join(models_dir, "ridge_model.pkl"), "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form["Temperature"])
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))  
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        new_data = scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge.predict(new_data)
        return render_template("home.html", result=result[0])
    else:
        return render_template("home.html")
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)