import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")

# Adjust the size of the figure
plt.figure(figsize=(10, 6))

# Adjust the thickness of the lines
plt.rcParams['lines.linewidth'] = 3
# Adjust the font size
plt.rcParams['font.size'] = 16

with open('recording_ILPDDQN_1000car_0124_THU_100percent_offline_DDQN.pkl', 'rb') as f:
    loss = pickle.load(f)
    total_reward_plot = pickle.load(f)
    contradiction = pickle.load(f)
    Detour = pickle.load(f)
    Validation = pickle.load(f)

with open('recording_ILPDDQN_1000car_0126_THU_100percent_offline_cql500.pkl', 'rb') as f3:
    loss1 = pickle.load(f3)
    total_reward_plot1 = pickle.load(f3)
    contradiction1 = pickle.load(f3)
    Detour1 = pickle.load(f3)
    Validation1 = pickle.load(f3)

with open('recording_ILPAC_1000car_0207_100percent_offline_thu_1.pkl', 'rb') as f3:
    loss2 = pickle.load(f3)
    total_reward_plot2 = pickle.load(f3)
    contradiction2 = pickle.load(f3)
    Detour2 = pickle.load(f3)
    Validation2 = pickle.load(f3)

s1 = pd.Series(Validation)

# Compute rolling statistics for s2
s2 = pd.Series(Validation1)

s3 = pd.Series(Validation2)

# Compute rolling statistics for s1
rolling_avg1 = s1.rolling(window=1).mean()
rolling_max1 = s1.rolling(window=1).max()
rolling_min1 = s1.rolling(window=1).min()


# 假设 s1 和 s2 是 Pandas Series
s1_length = len(s1)
s2_length = len(s2)

# 计算长度比例
ratio = s2_length / s1_length

# 根据比例计算s2的等比例索引
s2_indices = [round(i * ratio) for i in range(s1_length)]

# 从s2中取出等比例的元素
s2_resampled = s2.iloc[s2_indices]
s2_resampled = s2_resampled.reset_index(drop=True)

# 对等比例采样后的数据应用滚动窗口操作
rolling_avg2 = s2_resampled.rolling(window=1).mean()
rolling_max2 = s2_resampled.rolling(window=1).max()
rolling_min2 = s2_resampled.rolling(window=1).min()

rolling_avg3 = s3.rolling(window=1).mean()

# Plot rolling averages, maximums, and minimums for s3
plt.plot(rolling_avg1, label='ILPDDQN', color='blue')
plt.fill_between(range(len(rolling_avg1)), rolling_min1, rolling_max1, color='blue', alpha=0.1)

# Plot rolling averages, maximums, and minimums for s2
plt.plot(rolling_avg2, label='MA-BMCQL', color='purple')
plt.fill_between(range(len(rolling_avg2)), rolling_min2, rolling_max2, color='purple', alpha=0.1)

plt.plot(rolling_avg3, label='ILPAC', color='brown')

# Add a title and labels
plt.title('Comparisons in RS for task 1000_CARS_Thus')
plt.xlabel('Training Steps/10K')
plt.ylabel('Accumulative Total Rewards')

# Add a legend
plt.legend()

# Display the plot
plt.show()

