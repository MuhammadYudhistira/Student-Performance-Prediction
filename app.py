from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

app = Flask(__name__)
model = pickle.load(open("modelrf.pkl", "rb"))

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        gender = int(request.form.get('gender'))
        parent = int(request.form.get('parent'))
        lanche = int(request.form.get('lanche'))
        test_prep = int(request.form.get('test_prep'))
        math_score = int(request.form.get('math_score'))
        wrt_score = int(request.form.get('wrt_score'))
        X = np.array([gender, parent, lanche, test_prep, math_score, wrt_score]).reshape(1, -1)
        result = int(model.predict(X))
        persen = np.amax(model.predict_proba(X))*100
        if result== 1 :
            if persen <= 100:
                proba = "performa Siswa Baik sekali, Siswa Berkemungkinan Lulus dengan probabilitas Lulus " + str(persen)
            elif persen <= 50:
                proba = "performa Siswa Biasa, Siswa Berkemungkinan Lulus dengan probabilitas Lulus " + str(persen) 
        else:
            if persen >= 61:
                proba = "performa Siswa Buruk sekali, Siswa Berkemungkinan Gagal dengan probabilitas Gagal " + str(persen)
            elif persen <= 60:
                proba = "performa Siswa Biasa, Siswa Berkemungkinan Gagal dengan probabilitas Gagal " + str(persen)
        return render_template("index.html", result=result, proba=proba)
    
    else:
        return render_template("index.html")
    
if __name__ == "__main__":
    app.run(debug=True)