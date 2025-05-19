# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# %%
#step1
def calculate_volatility(data, n):
    x1 = data[:n]
    x_1 = np.sqrt(np.sum(x1)/n)
    sigma1 = np.sqrt(np.sum(x1 - x_1) / n)

    x2 = data[-n:]
    x_2 = np.sqrt(np.sum(x2)/n)
    sigma2 = np.sqrt(np.sum((x2 - x_2)) / n)

    return sigma1, sigma2

def add_padding(data, sigma1, sigma2, m):
    square_sigma1=np.sqrt(sigma1)
    square_sigma2=np.sqrt(sigma2)
    
    padding_front = []
    for i in range(m):
        if i == 0:
            padding_front.append(data[0] + np.random.normal(0, square_sigma1))
        else:
            padding_front.append(padding_front[-1] + np.random.normal(0, square_sigma1))

    padding_back = []
    for i in range(m):
        if i == 0:
            padding_back.append(data[-1] + np.random.normal(0, square_sigma2))
        else:
            padding_back.append(padding_back[-1] + np.random.normal(0, square_sigma2))

    new_data = np.concatenate([padding_front[::-1], data, padding_back])
    return new_data

# %%
#step2:FFT将时域信号转换为频域信号，并返回频率、振幅以及原始FFT结果
def perform_fft(data, use_shift=True):
    fft_result = np.fft.fft(data)#进行快速傅里叶变换，得到频域结果fft_result
    frequencies = np.fft.fftfreq(len(data), d=1)#频率数组 frequencies，采样间隔为 1
    amplitude = np.abs(fft_result)#振幅，即fft_result 的绝对值
    if use_shift:
        frequencies = np.fft.fftshift(frequencies)#将频率和振幅数组重新排列，使零频成分居中。
        amplitude = np.fft.fftshift(amplitude)
    
    return {#返回一个字典，包含频率数组、振幅数组和原始 FFT 结果
        'frequencies': frequencies,
        'amplitude': amplitude,
        'fft_result': fft_result
    }

# %%
# step3: 根据给定阈值对频域信号进行滤波
def apply_threshold(fft_data, epsilon,use_shift=True):
    frequencies = fft_data['frequencies']
    fft_result = fft_data['fft_result']
    # 创建一个与fft_result相同形状的全零数组
    filtered_fft_result = np.zeros_like(fft_result,dtype=complex)

    for k in range(len(frequencies)):
        f_k = frequencies[k]
        if np.abs(f_k) > np.abs(epsilon):
            # 如果频率绝对值小于阈值绝对值，保留原始系数
            filtered_fft_result[k] = fft_result[k]
        else:
            # 否则将系数设为0
            filtered_fft_result[k] = 0

        filtered_amplitude = np.abs(np.fft.fftshift(filtered_fft_result)) if use_shift else np.abs(filtered_fft_result)
    return {
        'frequencies': frequencies,
        'amplitude': filtered_amplitude,
        'fft_result': filtered_fft_result
    }


# %%
#step4：逆FFT变换恢复时间域
def inverse_fft(fft_filtered_data):
    # 假设输入是字典，提取 fft_result
    fft_filtered = fft_filtered_data['fft_result']
    denoised_padded = np.fft.ifft(fft_filtered).real  # 取实部，因为理论上虚部应该很小
    return denoised_padded

# %%
#step5：移除填充区域
def remove_padding(denoised_padded, original_length, m):
    start_idx = m
    end_idx = m + original_length
    denoised_data = denoised_padded[start_idx:end_idx]
    return denoised_data

# %%
#step1可视化-折线图
def plot_data(data, dates,title='Title', xlabel='Year', ylabel='Y-axis', label='Data'):
    plt.figure(figsize=(5, 4))
    plt.plot(dates,data, color='blue', label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# %%
#step2可视化-折线图2
def plot_padded_data(new_data, m, stock_data,dates):
    # 分离填充和原始数据
    padding_front = new_data[:m]
    original_data = new_data[m:m + len(stock_data)]
    padding_back = new_data[m + len(stock_data):]

    plt.figure(figsize=(5, 4))

# 忽略 dates，直接绘制（用默认 index）
    plt.plot(range(m), padding_front, color='red', label='Front Padding')
    plt.plot(range(m, m + len(original_data)), original_data, color='blue', label='Original Data')
    plt.plot(range(m + len(original_data), len(new_data)), padding_back, color='red', label='Back Padding')

# 不显示横坐标刻度
    plt.xticks([])  # 清除x轴刻度
    plt.xlabel('')  # 可选：移除x轴标签文字
    plt.ylabel('Value')
    plt.title('Stock Data with Padding')

    plt.legend()
    plt.tight_layout()
    plt.gca().invert_xaxis()
    plt.show()

# %%
# step3可视化-谱线图
def visualize_fft_results(fft_data, title="FFT Result", inset_range=None, figsize=(6, 4),ylim=None):

    # 从输入的数据里提取出频率值和对应的振幅值，准备用来画图
    frequencies, amplitude = fft_data['frequencies'], fft_data['amplitude']
    
    # 创建一个画布(fig)和主图区域(ax)，大小由figsize参数决定
    fig, ax = plt.subplots(figsize=figsize)

    # 在主图的右上角创建一个小图(axins)，大小是主图的35%宽和35%高
    axins = inset_axes(ax, width="35%", height="35%", loc='upper right')
    
    # 同时在主图和小图上绘制频率-振幅曲线(蓝色线条)
    # 并且都加上网格线，透明度设为0.3(半透明效果)
    for a in [ax, axins]:
        a.plot(frequencies, amplitude, 'b-')
        a.grid(True, alpha=0.3)
    
    # 设置主图属性
    # 给主图加上X轴标签(频率)、Y轴标签(振幅)和标题(由title参数决定)
    ax.set(xlabel='Frequency (Hz)', 
           ylabel='Amplitude', 
           title=title)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    
    # 设置小图属性
    if inset_range:
        # 如果指定了inset_range参数，就设置小图的X轴和Y轴显示范围
        # 同时给小图加上标题"High Frequency Details"(高频细节)
        axins.set(xlim=inset_range[:2], ylim=inset_range[2:], 
                  title='High Frequency Details')
        
        # 在主图和小图之间画一个指示框和连接线
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # 自动调整子图的位置和大小
    plt.tight_layout()
    plt.show()


# %%
# %%
# 完整流程复现（以AAPL数据为例）
file_path = 'SP500.csv'
stock_data = pd.read_csv(file_path)
close_prices = stock_data['close'].values
dates = pd.to_datetime(stock_data['date'])  # 确保日期数据为datetime类型

N = len(close_prices)
m = 40  # 填充区域大小
n = 20  # 波动率计算窗口
epsilon = 0.3  # 频率阈值

# 步骤1：计算波动率并添加填充
sigma1, sigma2 = calculate_volatility(close_prices, n)
new_data = add_padding(close_prices, sigma1, sigma2, m)

# 步骤2：FFT变换
fft_data = perform_fft(new_data,use_shift=True)

# 步骤3：阈值处理
filtered_fft_data= apply_threshold(fft_data, epsilon,use_shift=True)
amplitude_filtered = filtered_fft_data['amplitude']

# 步骤4：逆FFT恢复时间域
denoised_padded = inverse_fft(filtered_fft_data)

# 步骤5：移除填充区域
denoised_data = remove_padding(denoised_padded, N, m)


# 可视化1原始数据
plot_data(close_prices, dates, title='Stock Data', xlabel='Date', ylabel='Value', label='Original Data')

# 可视化2填充后的数据
plot_padded_data(new_data, m, close_prices, dates)



# 可视化3：FFT频谱
visualize_fft_results(
  fft_data,
  title="Original FFT Spectrum",
  inset_range=(0, 0.4, 0, 300) ,
  ylim=(0,70000)
)

#可视化4：过滤后的频谱
visualize_fft_results(
  filtered_fft_data ,
  title=" Filtered FFT Spectrum",
  inset_range=(0, 0.4, 0, 300),
  ylim=(0,70000)
)

# # 可视化5：过滤后的时域
# plt.figure(figsize=(10, 5))
# date_range = pd.date_range(start=dates.iloc[0] - pd.Timedelta(days=m), periods=len(new_data), freq='D')
# # 绘制填充后的原始数据（蓝色）
# plt.plot(date_range, new_data, 'b-', alpha=0.6, label='Padded Original Data')
# # 绘制去噪后的填充数据（红色），修正了格式化问题
# plt.plot(date_range, denoised_padded, 'r-', label=f'Denoised Padded Data (epsilon={epsilon})')
# plt.title('Padded Data Comparison')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

# 可视化6：去掉之前的填充部分
plt.figure(figsize=(10, 5))
# 绘制原始数据（蓝色）
plt.plot(dates, close_prices, 'b-', alpha=0.6, label='Original Data')
# 绘制最终去噪数据（红色），修正了格式化问题
plt.plot(dates, denoised_data, 'r-', label=f'Final Denoised Data (epsilon={epsilon})')
plt.title('Original vs Final Denoised Data')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


