import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from p_ftd import calculate_volatility, add_padding, perform_fft, apply_threshold, inverse_fft, remove_padding

def p_ftd_denoise_pipeline(data_array, N=40, m=40, epsilon=0.2):
    sigma1, sigma2 = calculate_volatility(data_array, N)
    padded = add_padding(data_array, sigma1, sigma2, m)
    fft_result = perform_fft(padded)
    filtered_fft = apply_threshold(fft_result, epsilon)
    denoised_padded = inverse_fft(filtered_fft)
    denoised = remove_padding(denoised_padded, len(data_array), m)
    return denoised

# 1. 读取S&P500数据（使用多特征）
df = pd.read_csv('SP500.csv', parse_dates=['date'])
df.sort_values('date', inplace=True)

features = ['open', 'high', 'low', 'close', 'volume']
target_col = 'close'

data = df[features].values  # shape: (n_samples, 5)
target = df[target_col].values  # 用于拟合和评估
dates = df['date']
n_samples = len(data)

# 2. 对 close 列使用 P-FTD 去噪
denoised_close = p_ftd_denoise_pipeline(target, N=40, m=40, epsilon=0.2)
data_denoised = data.copy()
data_denoised[:, 3] = denoised_close

# 替换掉原数据中的 close 列（列索引为3）
data_denoised = data.copy()
data_denoised[:, 3] = denoised_close

# 3. 全特征归一化（统一 MinMax）
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
scaled_denoised = scaler.transform(data_denoised)


# 构造监督学习数据集
def create_dataset(multivariate_data, time_step=20, target_index=3):
    X, y = [], []
    for i in range(len(multivariate_data) - time_step - 1):
        X.append(multivariate_data[i:(i + time_step), :])  # 多特征输入
        y.append(multivariate_data[i + time_step, target_index])  # 预测未来close
    return np.array(X), np.array(y)

time_step = 20
X_raw, y_raw = create_dataset(scaled_data, time_step)
X_denoised, y_denoised = create_dataset(scaled_denoised, time_step)

# 4. 构建RNN模型
def build_rnn_model(input_dim=5):  # 多特征输入
    model = Sequential()
    model.add(SimpleRNN(10, input_shape=(time_step, input_dim)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# 5. 模型训练
train_size = int(0.7 * n_samples)
val_size = int(0.1 * n_samples)

model_denoised = build_rnn_model(input_dim=X_denoised.shape[2])
model_denoised.fit(X_denoised[:train_size], y_denoised[:train_size],
                   validation_data=(X_denoised[train_size:train_size + val_size], y_denoised[train_size:train_size + val_size]),
                   epochs=100, batch_size=32, verbose=0)

model_raw = build_rnn_model(input_dim=X_raw.shape[2])
model_raw.fit(X_raw[:train_size], y_raw[:train_size],
              validation_data=(X_raw[train_size:train_size + val_size], y_raw[train_size:train_size + val_size]),
              epochs=100, batch_size=32, verbose=0)

# 6. 模型评估
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test, verbose=0)
    # 只对 close 特征反归一化（对应原数据第3列）
    dummy = np.zeros((len(y_test), scaled_data.shape[1]))
    dummy[:, 3] = y_test
    y_test_inv = scaler.inverse_transform(dummy)[:, 3]

    dummy[:, 3] = y_pred.flatten()
    y_pred_inv = scaler.inverse_transform(dummy)[:, 3]

    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    return mae, rmse

# 测试集评估
X_test_denoised = X_denoised[train_size + val_size:]
y_test_denoised = y_denoised[train_size + val_size:]
X_test_raw = X_raw[train_size + val_size:]
y_test_raw = y_raw[train_size + val_size:]

mae_denoised, rmse_denoised = evaluate_model(model_denoised, X_test_denoised, y_test_denoised)
mae_raw, rmse_raw = evaluate_model(model_raw, X_test_raw, y_test_raw)

print(f"去噪数据 - MAE: {mae_denoised:.4f}, RMSE: {rmse_denoised:.4f}")
print(f"原始数据 - MAE: {mae_raw:.4f}, RMSE: {rmse_raw:.4f}")


# ������ - MAE: 4.8731, RMSE: 6.3578
# ԭʼ���� - MAE: 6.5600, RMSE: 8.6546