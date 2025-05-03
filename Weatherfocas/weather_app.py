import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)

# OpenWeatherMap API key
API_KEY = "532396ff2ce38dedbaa9c77f53214b7a"
WEATHER_URL = "http://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "http://api.openweathermap.org/data/2.5/forecast"

# Hàm lấy dữ liệu thời tiết hiện tại
def get_weather(city):
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    response = requests.get(WEATHER_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        return {
            "city": data["name"],
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "lat": data["coord"]["lat"],
            "lon": data["coord"]["lon"]
        }
    return None

# Hàm lấy dự báo 5 ngày
def get_forecast(city):
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    response = requests.get(FORECAST_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        return [{
            "temp": item["main"]["temp"],
            "humidity": item["main"]["humidity"],
            "description": item["weather"][0]["description"],
            "dt_txt": item["dt_txt"]
        } for item in data["list"][:8]]  # Lấy 8 điểm dữ liệu (24 giờ)
    return []

# Hàm xây dựng mô hình LSTM
def build_lstm_model():
    model = Sequential([
        Input(shape=(10, 2)),
        LSTM(100, activation='relu', return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(2)  # Dự đoán nhiệt độ và độ ẩm
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    return model

# Tải hoặc tạo mô hình LSTM
model_path = 'lstm_weather_model.keras'
scaler_path = 'scaler.npy'
model = None
scaler = None

try:
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        scaler = np.load(scaler_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading model or scaler: {e}")

if model is None or scaler is None:
    scaler = MinMaxScaler()
    X = np.random.rand(1000, 10, 2) * 30
    y = X[:, -1, :] + np.random.rand(1000, 2) * 2
    X_scaled = scaler.fit_transform(X.reshape(-1, 2)).reshape(X.shape)
    y_scaled = scaler.transform(y)

    model = build_lstm_model()
    model.fit(X_scaled, y_scaled, epochs=20, batch_size=32, verbose=0)
    model.save(model_path)
    np.save(scaler_path, scaler)

# Hàm dự đoán
def predict_weather(historical_data):
    data = np.array(historical_data)
    data_scaled = scaler.transform(data.reshape(-1, 2)).reshape(1, 10, 2)
    prediction_scaled = model.predict(data_scaled, verbose=0)
    prediction = scaler.inverse_transform(prediction_scaled)
    return {"temp": float(prediction[0][0]), "humidity": float(prediction[0][1])}

@app.route("/weather", methods=["POST"])
def weather():
    data = request.get_json()
    city = data.get("city")
    if not city:
        return jsonify({"error": "City is required"}), 400
    
    weather_data = get_weather(city)
    forecast_data = get_forecast(city)
    
    if weather_data:
        historical_data = [[weather_data["temp"] - i*0.5, weather_data["humidity"] - i*0.5] for i in range(10)]
        prediction = predict_weather(historical_data)
        return jsonify({
            "current": weather_data,
            "forecast": forecast_data,
            "prediction": prediction
        })
    return jsonify({"error": "City not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)