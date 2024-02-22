# 读取csv文件
import pandas as pd

# 读取csv文件
df = pd.read_csv('csv\demand_new.csv')

# 随机抽取10%的订单
sample_df = df.sample(frac=0.3)

# 按照原有顺序排序
sample_df = sample_df.sort_index()

# 创建一个布尔序列，表示 'minute' 列是否在 481 和 510（含）之间
mask = (sample_df['minute'] >= 481) & (sample_df['minute'] <= 510)

# 使用该布尔序列来筛选出所需的行
filtered_df = sample_df[mask]

# 打印筛选出的行数
print(len(filtered_df))

# 写入新的csv文件
sample_df.to_csv('csv\demand_new_sample_1.csv', index=False)