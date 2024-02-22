import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Adjust the size of the figure
plt.figure(figsize=(10, 6))

# Adjust the thickness of the lines
plt.rcParams['lines.linewidth'] = 3
# Adjust the font size
plt.rcParams['font.size'] = 16

with open('recording_ILPDDQN_1000car_1227_WED_100percent_005tau.pkl', 'rb') as f:
    loss = pickle.load(f)
    total_reward_plot = pickle.load(f)
    contradiction = pickle.load(f)
    Detour = pickle.load(f)
    Validation = pickle.load(f)


s1 = pd.Series(total_reward_plot)

# Compute rolling statistics for s1
rolling_avg1 = s1.rolling(window=100).mean()
rolling_max1 = s1.rolling(window=100).max()
rolling_min1 = s1.rolling(window=100).min()


# Plot rolling averages, maximums, and minimums for s1
plt.plot(rolling_avg1, label='ILPDDQN', color='blue')
plt.fill_between(range(len(rolling_avg1)), rolling_min1, rolling_max1, color='blue', alpha=0.1)

# Add a title and labels
plt.title('Online Training Curve of ILPDDQN for task WED_1000cars')
plt.xlabel('Episodes')
plt.ylabel('Accumulative Total Rewards')

# Add a legend
plt.legend()

# Display the plot
plt.show()

