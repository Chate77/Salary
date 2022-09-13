from flask import Flask,jsonify,request
import pandas as pd
import numpy as np
import pickle
app=Flask(__name__)

lr_model=pickle.load(open("lr.pkl","rb"))
column_list=pickle.load(open("list77.pkl","rb"))

@app.route('/')
def hello():
    return "hello guys welcome my world"

@app.route("/prediction")
def predicton():
    data = request.get_json()
    Age=data['Age']
    Weight=data['Weight']
    Experience=data['Experience']

    lr_dt={"Age":[Age],"Weight":[Weight],"Experience":[Experience]}

    test_df=pd.DataFrame(data=lr_dt)
    prediction = lr_model.predict(test_df)

    return jsonify({"prediction":prediction[0],
                "Age":Age,
                "Weight":Weight,
                "Experience":Experience})

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5006)

