import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Giả lập dữ liệu lịch sử
def generate_dummy_data(samples=1000, timesteps=10):
    data = np.random.rand(samples, timesteps, 2) * 30  # Nhiệt độ, độ ẩm
    target = data[:, -1, :] + np.random.rand(samples, 2) * 2
    return data, target

# Tiền xử lý dữ liệu
scaler = MinMaxScaler()
def preprocess_data(data):
    return scaler.fit_transform(data.reshape(-1, 2)).reshape(data.shape)

# Xây dựng mô hình
def build_lstm_model():
    model = Sequential([
        LSTM(100, activation='relu', input_shape=(10, 2), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Huấn luyện mô hình
X, y = generate_dummy_data()
X = preprocess_data(X)
y = scaler.fit_transform(y)

model = build_lstm_model()
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Lưu mô hình và scaler
model.save('lstm_weather_model.h5')
np.save('scaler.npy', scaler)