from flask import Flask, render_template, request,redirect
import numpy as np
import pandas as pd
import joblib


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    return render_template('index.html')

#---------------------------------------------------------------------------#


model = joblib.load('random_forest_model.joblib')

@ app.route('/price_prediction', methods=['POST'])
def price_prediction():
    if request.method == 'POST':
        age_str = request.form['age']
        sex = request.form['sex']
        bmi_str = request.form['bmi']
        children_str = request.form['children']
        smoker = request.form['smoker']
        region = request.form['region']

        #1. preprocess
        #print(age, sex, bmi,children,smoker,region)
        age = int(age_str)
        bmi = float(bmi_str)
        children = int(children_str)
        if sex == 'male':
            sex = 0
        else:
            sex = 1

        if smoker == 'yes':
            smoker=0
        else:
            smoker=1

        if region == 'southeast':
            region=0
        elif region == 'southwest':
            region=1
        elif region == 'northeast':
            region =2
        else:
            region =3 
        
        #print(age, sex, bmi,children,smoker,region)

        input_data = np.array([age, sex, bmi,children,smoker,region]+ [0] * (53 - 6)).reshape(1, -1)
        #print(input_data)

        # changing input_data to a numpy array
        #input_data_as_numpy_array = input_data.reshape(1,-1)
        #print(input_data_as_numpy_array)


        prediction = model.predict(input_data)
        #print(prediction)
        
        return render_template('result.html', prediction= np.round(prediction[0]))
    

    
    

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
