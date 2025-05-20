# %%
# ✅ Step 1: 导入依赖项
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


# %%
# ✅ Step 2: 设置随机种子（确保可复现）
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)


# %%
# ✅ Step 3: 读取数据并对齐文献时间
df = pd.read_csv("SP500.csv", parse_dates=["date"])
df.sort_values("date", inplace=True)

df = df[["date", "open", "high", "low", "close", "volume"]]
df.reset_index(drop=True, inplace=True)

features = ['open', 'high', 'low', 'close', 'volume']
target_col = 'close'
target = df[target_col].values
dates = df['date']


# %%
# ✅ Step 4: 定义 P-FTD
def calculate_volatility(data, n):
    x1 = data[:n]
    x_1 = np.sqrt(np.sum(x1) / n)
    sigma1 = np.sqrt(np.sum(x1 - x_1) / n)

    x2 = data[-n:]
    x_2 = np.sqrt(np.sum(x2) / n)
    sigma2 = np.sqrt(np.sum((x2 - x_2)) / n)

    return sigma1, sigma2

def add_padding(data, sigma1, sigma2, m):
    padding_front = [data[0] + np.random.normal(0, np.sqrt(sigma1))]
    for _ in range(1, m):
        padding_front.append(padding_front[-1] + np.random.normal(0, np.sqrt(sigma1)))

    padding_back = [data[-1] + np.random.normal(0, np.sqrt(sigma2))]
    for _ in range(1, m):
        padding_back.append(padding_back[-1] + np.random.normal(0, np.sqrt(sigma2)))

    return np.concatenate([padding_front[::-1], data, padding_back])

def perform_fft(data):
    fft_result = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(data), d=1)
    amplitude = np.abs(np.fft.fftshift(fft_result))
    return {'frequencies': frequencies, 'amplitude': amplitude, 'fft_result': fft_result}

def apply_threshold(fft_data, epsilon):
    frequencies = fft_data['frequencies']
    fft_result = fft_data['fft_result']
    filtered_fft_result = np.zeros_like(fft_result, dtype=complex)
    for k in range(len(frequencies)):
        if np.abs(frequencies[k]) > epsilon:
            filtered_fft_result[k] = fft_result[k]
    return {'frequencies': frequencies, 'amplitude': np.abs(filtered_fft_result), 'fft_result': filtered_fft_result}

def inverse_fft(fft_filtered_data):
    return np.fft.ifft(fft_filtered_data['fft_result']).real

def remove_padding(denoised_padded, original_length, m):
    return denoised_padded[m:m + original_length]


# %%
# ✅ Step 5: 应用 P-FTD 去噪（N=40, m=40, ε=0.2）
n, m, epsilon = 40, 40, 0.2
sigma1, sigma2 = calculate_volatility(target, n)
padded_data = add_padding(target, sigma1, sigma2, m)
fft_data = perform_fft(padded_data)
filtered_fft_data = apply_threshold(fft_data, epsilon)
denoised_padded = inverse_fft(filtered_fft_data)
denoised_data = remove_padding(denoised_padded, len(target), m)


# %%
# ✅ Step 6: 构造滑动窗口数据
def create_sequences(data, lookback=20):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)


# %%
# ✅ Step 7: 定义模型训练 + 还原误差计算
def evaluate_model(data, model_type='LSTM', lookback=20):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.reshape(-1, 1))

    X, y = create_sequences(scaled, lookback)
    total_len = len(X)
    train_end = int(total_len * 0.7)
    val_end = int(total_len * 0.8)

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    X_train = X_train.reshape((-1, lookback, 1))
    X_val = X_val.reshape((-1, lookback, 1))
    X_test = X_test.reshape((-1, lookback, 1))

    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(10, input_shape=(lookback, 1)))
    else:
        model.add(GRU(10, input_shape=(lookback, 1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, batch_size=32, verbose=0,
              callbacks=tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True))

    y_pred = model.predict(X_test)
    y_pred_inv = scaler.inverse_transform(y_pred).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    return mae, rmse


# %%
# ✅ Step 8: 比较四个模型的表现
mae_lstm, rmse_lstm = evaluate_model(target, 'LSTM')
mae_gru, rmse_gru = evaluate_model(target, 'GRU')
mae_pftd_lstm, rmse_pftd_lstm = evaluate_model(denoised_data, 'LSTM')
mae_pftd_gru, rmse_pftd_gru = evaluate_model(denoised_data, 'GRU')


# %%
# ✅ Step 9: 输出最终对比结果
print("S&P500预测性能对比（2001–2020）:")
print(f"{'模型':<20} {'MAE':>10} {'RMSE':>10}")
print("-" * 40)
print(f"{'LSTM':<20} {mae_lstm:10.4f} {rmse_lstm:10.4f}")
print(f"{'GRU':<20} {mae_gru:10.4f} {rmse_gru:10.4f}")
print(f"{'P-FTD_LSTM':<20} {mae_pftd_lstm:10.4f} {rmse_pftd_lstm:10.4f}")
print(f"{'P-FTD_GRU':<20} {mae_pftd_gru:10.4f} {rmse_pftd_gru:10.4f}")



