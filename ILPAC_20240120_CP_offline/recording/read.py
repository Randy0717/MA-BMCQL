import pickle
import pandas as pd
import matplotlib.pyplot as plt

with open('recording_ILPAC_1000car_0207_100percent_offline_fri_1.pkl', 'rb') as f:
    loss = pickle.load(f)
    total_reward_plot = pickle.load(f)
    contradiction = pickle.load(f)
    Detour = pickle.load(f)
    Validation = pickle.load(f)

s1 = pd.Series(loss)
rolling_avg1 = s1.rolling(window=1000).mean()
rolling_max1 = s1.rolling(window=1000).max()
rolling_min1 = s1.rolling(window=1000).min()
rolling_avg_list1 = rolling_avg1.tolist()
rolling_max_list1 = rolling_max1.tolist()
rolling_min_list1 = rolling_min1.tolist()

s2 = pd.Series(total_reward_plot)
rolling_avg2 = s2.rolling(window=100).mean()
rolling_max2 = s2.rolling(window=100).max()
rolling_min2 = s2.rolling(window=100).min()
rolling_avg_list2 = rolling_avg2.tolist()
rolling_max_list2 = rolling_max2.tolist()
rolling_min_list2 = rolling_min2.tolist()

s3 = pd.Series(contradiction)
rolling_avg3 = s3.rolling(window=1).mean()
# rolling_max3 = s3.rolling(window=50).max()
# rolling_min3 = s3.rolling(window=50).min()
rolling_avg_list3 = rolling_avg3.tolist()
# rolling_max_list3 = rolling_max3.tolist()
# rolling_min_list3 = rolling_min3.tolist()

s4 = pd.Series(Validation)
rolling_avg4 = s4.rolling(window=1).mean()
# rolling_max4 = s4.rolling(window=1000).max()
# rolling_min4 = s4.rolling(window=1000).min()
rolling_avg_list4 = rolling_avg4.tolist()
# rolling_max_list4 = rolling_max4.tolist()
# rolling_min_list4 = rolling_min4.tolist()

# Create a figure with 2 subplots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
# Plot the first graph on the first subplot
ax1.plot(rolling_avg_list1)
# ax1.plot(rolling_max_list1)
# ax1.plot(rolling_min_list1)
# ax1.fill_between(range(len(rolling_min_list1)), rolling_min_list1, rolling_max_list1, alpha=0.5)
ax1.set_xlabel('training steps')
ax1.set_ylabel('training loss')
ax1.set_title('loss')

# Plot the second graph on the second subplot
ax2.plot(rolling_avg_list2)
ax2.plot(rolling_max_list2)
ax2.plot(rolling_min_list2)
ax2.fill_between(range(len(rolling_min_list2)), rolling_min_list2, rolling_max_list2, alpha=0.5)
ax2.set_xlabel('episodes')
ax2.set_ylabel('total reward')
ax2.set_title('total reward')

# Plot the third graph on the second subplot
ax3.plot(rolling_avg_list3)
# ax3.plot(rolling_max_list3)
# ax3.plot(rolling_min_list3)
# ax3.fill_between(range(len(rolling_min_list3)), rolling_min_list3, rolling_max_list3, alpha=0.5)
ax3.set_xlabel('episodes')
ax3.set_ylabel('contradiction rate')
ax3.set_title('Contradiction')

ax4.plot(rolling_avg_list4)
# ax4.plot(rolling_max_list4)
# ax4.plot(rolling_min_list4)
# ax4.fill_between(range(len(rolling_min_list4)), rolling_min_list4, rolling_max_list4, alpha=0.5)
ax4.set_xlabel('episodes')
ax4.set_ylabel('Rewards')
ax4.set_title('Validation')

plt.show()