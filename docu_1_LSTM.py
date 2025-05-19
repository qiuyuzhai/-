import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 设置随机种子以确保结果可复现
np.random.seed(42)
tf.random.set_seed(42)

# 从外部模块导入P-FTD去噪函数
from p_ftd import calculate_volatility, add_padding, perform_fft, apply_threshold, inverse_fft, remove_padding

# ----------- 1. 数据读取与预处理 ----------------
df = pd.read_csv('SP500.csv', parse_dates=['date'])
df.sort_values('date', inplace=True)

features = ['open', 'high', 'low', 'close', 'volume']
target_col = 'close'

data = df[features].values  # shape: (n_samples, 5)
target = df[target_col].values
dates = df['date']
n_samples = len(data)

# 文献默认参数
N = 40
m = 40
epsilon = 0.2

# ----------- 2. 使用 P-FTD 去噪 close 列 ----------------
def p_ftd_denoise_pipeline(data_array, N, m, epsilon):
    sigma1, sigma2 = calculate_volatility(data_array, N)
    padded = add_padding(data_array, sigma1, sigma2, m)
    fft_result = perform_fft(padded)
    filtered_fft = apply_threshold(fft_result, epsilon)
    denoised_padded = inverse_fft(filtered_fft)
    denoised = remove_padding(denoised_padded, len(data_array), m)
    return denoised

denoised_close = p_ftd_denoise_pipeline(target, N, m, epsilon)

# 替换原数据中的 close
data_denoised = data.copy()
data_denoised[:, 3] = denoised_close

# ----------- 3. 归一化：多特征整体归一化 ----------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
scaled_denoised = scaler.transform(data_denoised)

# ----------- 4. 构造时间序列数据 ----------------
time_step = 20  # 文献设定为20天（约一个月）

def create_dataset(multivariate_data, time_step=20, target_index=3):
    X, y = [], []
    for i in range(len(multivariate_data) - time_step - 1):
        X.append(multivariate_data[i:(i + time_step), :])
        y.append(multivariate_data[i + time_step, target_index])
    return np.array(X), np.array(y)

X_raw, y_raw = create_dataset(scaled_data, time_step)
X_denoised, y_denoised = create_dataset(scaled_denoised, time_step)

# ----------- 5. 划分数据集 ----------------
total_samples = X_raw.shape[0]
train_size = int(0.7 * total_samples)
val_size = int(0.1 * total_samples)

X_train_raw, y_train_raw = X_raw[:train_size], y_raw[:train_size]
X_val_raw, y_val_raw = X_raw[train_size:train_size+val_size], y_raw[train_size:train_size+val_size]
X_test_raw, y_test_raw = X_raw[train_size + val_size:], y_raw[train_size + val_size:]

X_train_denoised, y_train_denoised = X_denoised[:train_size], y_denoised[:train_size]
X_val_denoised, y_val_denoised = X_denoised[train_size:train_size+val_size], y_denoised[train_size:train_size+val_size]
X_test_denoised, y_test_denoised = X_denoised[train_size + val_size:], y_denoised[train_size + val_size:]

# ----------- 6. 构建LSTM模型 ----------------
def build_lstm_model(input_shape, dropout_rate=0.2):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(64, return_sequences=False),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ----------- 7. 模型训练与评估 ----------------
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, scaler, target_index=3):
    # 设置回调函数
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # 训练模型
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # 预测
    y_pred_scaled = model.predict(X_test).flatten()
    
    # 反归一化
    # 注意：scaler是对所有特征进行的归一化，需要重构数据才能正确反归一化
    y_test_reshaped = np.zeros((len(y_test), scaler.n_features_in_))
    y_test_reshaped[:, target_index] = y_test
    
    y_pred_reshaped = np.zeros((len(y_pred_scaled), scaler.n_features_in_))
    y_pred_reshaped[:, target_index] = y_pred_scaled
    
    y_test_original = scaler.inverse_transform(y_test_reshaped)[:, target_index]
    y_pred_original = scaler.inverse_transform(y_pred_reshaped)[:, target_index]
    
    # 计算评估指标
    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    
    return mae, rmse, y_test_original, y_pred_original

# ----------- 8. 训练模型并获取结果 ----------------
input_shape = (time_step, X_raw.shape[2])

# 训练原始数据的LSTM模型
lstm_model_raw = build_lstm_model(input_shape)
mae_raw, rmse_raw, y_test_raw_original, y_pred_raw_original = train_and_evaluate_model(
    lstm_model_raw,
    X_train_raw, y_train_raw,
    X_val_raw, y_val_raw,
    X_test_raw, y_test_raw,
    scaler
)

# 训练P-FTD去噪数据的LSTM模型 (P-FTD_LSTM)
lstm_model_denoised = build_lstm_model(input_shape)
mae_denoised, rmse_denoised, y_test_denoised_original, y_pred_denoised_original = train_and_evaluate_model(
    lstm_model_denoised,
    X_train_denoised, y_train_denoised,
    X_val_denoised, y_val_denoised,
    X_test_denoised, y_test_denoised,
    scaler
)

# ----------- 9. 输出结果 ----------------
print(f"原始数据 - MAE: {mae_raw:.6f}, RMSE: {rmse_raw:.6f}")
print(f"去噪数据 - MAE: {mae_denoised:.6f}, RMSE: {rmse_denoised:.6f}")

# ԭʼ���� - MAE: 4.736339, RMSE: 6.232378
# ȥ������ - MAE: 5.118575, RMSE: 6.944864