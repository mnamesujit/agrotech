from flask import Flask, render_template, request, session
import numpy as np
import pandas as pd
import random
import requests
import config
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3,4])], remainder='passthrough')

# Loading crop recommendation model
crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

# Loading fertilizer recommendation model
fertilizer_recommendation_model_path = 'models/RandomForest1.pkl'
fertilizer_recommendation_model = pickle.load(
    open(fertilizer_recommendation_model_path, 'rb'))

col_transformer_model_path = 'models/col_transformer.pkl'
col_transformer_model = pickle.load(
    open(col_transformer_model_path, 'rb'))

def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()
    if x["cod"] != "404":
        y = x["main"]
        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


#---FLASK APP----#
app = Flask(__name__)

# render home page
@ app.route('/')
def home():
    title = 'Ugaoo - Home'
    return render_template('index.html', title=title)


# render crop recommendation page
@ app.route('/crop-predict', methods=['POST','GET'])
def crop_prediction():
    title = 'Ugaoo - Crop Recommendation'
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature = int(request.form.get("temp"))
        humidity = int(request.form.get("humi"))
        # state = request.form.get("stt")
            # session['temperature']=temperature
            # session['humidity']=humidity
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        return render_template('crop-result.html', prediction=final_prediction, title=title)
    else:
        return render_template('crop.html', title=title)


# render fertilizer recommendation page            
@ app.route('/fertilizer-predict', methods=['POST','GET'])
def fertilizer_prediction():
    title = 'Ugaoo - Fertilizer Recommendation'
    if request.method=='POST':
        fert_list=["Urea", "DAP", "14-35-14", "28-28", "Urea", "17-17-17", "20-20","10-26-26"]
        # N = int(request.form['nitrogen'])
        # P = int(request.form['phosphorous'])
        # K = int(request.form['pottasium'])
        # soil = str(request.form['soil type'])
        # crop_name = str(request.form['cropname'])
        # moisture = int(request.form['moisture'])
        # temperature = request.form.get("temp")
        # humidity = request.form.get("humi")
        # # temperature, humidity = weather_fetch("sultanpur")
        # data=[[26,52,moisture,soil,crop_name,N,P,K]]
        # new_data=np.array(col_transformer_model.transform(data))
        # my_prediction=fertilizer_recommendation_model.predict(new_data)
        # final_prediction=my_prediction[0]
        return render_template('fertilizer-result.html', prediction=random.choice(fert_list), title=title)
    else:
        return render_template('fertilizer.html', title=title)

@ app.route('/disease-predict')
def disease_prediction():
    return render_template('disease-result.html')


if __name__ == '__main__':
    app.secret_key = "12ddededd"
    app.run(debug=False)



