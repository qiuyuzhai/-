import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from p_ftd import calculate_volatility, add_padding, perform_fft, apply_threshold, inverse_fft, remove_padding
import matplotlib.pyplot as plt


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
#后来想用网格搜索法找最优参数，但是太耗时间了就放弃了

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

X_test_raw = X_raw[train_size + val_size:]
y_test_raw = y_raw[train_size + val_size:]
X_test_denoised = X_denoised[train_size + val_size:]
y_test_denoised = y_denoised[train_size + val_size:]

# ----------- 6. 构建模型（SimpleRNN） ----------------
def build_rnn_model(input_dim):
    model = Sequential()
    model.add(SimpleRNN(10, input_shape=(time_step, input_dim)))  # 文献用 hidden_size = 10
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# ----------- 7. 训练模型 ----------------
model_raw = build_rnn_model(input_dim=X_raw.shape[2])
model_raw.fit(X_raw[:train_size], y_raw[:train_size],
              validation_data=(X_raw[train_size:train_size + val_size], y_raw[train_size:train_size + val_size]),
              epochs=100, batch_size=32, verbose=0)

model_denoised = build_rnn_model(input_dim=X_denoised.shape[2])
model_denoised.fit(X_denoised[:train_size], y_denoised[:train_size],
                   validation_data=(X_denoised[train_size:train_size + val_size], y_denoised[train_size:train_size + val_size]),
                   epochs=100, batch_size=32, verbose=0)

# ----------- 8. 评估函数 ----------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test, verbose=0)

    # 只对 close 列反归一化
    dummy = np.zeros((len(y_test), scaled_data.shape[1]))
    dummy[:, 3] = y_test
    y_test_inv = scaler.inverse_transform(dummy)[:, 3]

    dummy[:, 3] = y_pred.flatten()
    y_pred_inv = scaler.inverse_transform(dummy)[:, 3]

    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    return mae, rmse, y_test_inv, y_pred_inv

# ----------- 9. 输出结果 ----------------
mae_raw, rmse_raw, y_true_raw, y_pred_raw = evaluate_model(model_raw, X_test_raw, y_test_raw)
mae_denoised, rmse_denoised, y_true_denoised, y_pred_denoised = evaluate_model(model_denoised, X_test_denoised, y_test_denoised)

print(f"原始数据 - MAE: {mae_raw:.4f}, RMSE: {rmse_raw:.4f}")
print(f"去噪数据 - MAE: {mae_denoised:.4f}, RMSE: {rmse_denoised:.4f}")

# ԭʼ���� - MAE: 7.4091, RMSE: 10.3192
# ȥ������ - MAE: 7.1851, RMSE: 8.8408