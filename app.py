import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
from numpy.core.numeric import outer
import pandas as pd

app=Flask(__name__)
regmodel=pickle.load(open('regression.pkl','rb'))
Scaler =pickle.load(open('scaling.pkl','rb'))
@app.route('/')

def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values()).reshape(1,-1)))
    new_data=Scaler.transform(np.array(list(data.values()).reshape(1,-1)))
    output= regmodel.predict(new_data)
    print(output[0])
    return jsonify(outer[0])

if __name__=='__main__':
    app.run(debug=True)