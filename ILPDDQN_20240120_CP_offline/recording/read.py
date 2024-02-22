import pickle
import pandas as pd
import matplotlib.pyplot as plt

with open('recording_ILPDDQN_1000car_0120_WED_100percent_offline.pkl', 'rb') as f:
    loss = pickle.load(f)
    total_reward_plot = pickle.load(f)
    contradiction = pickle.load(f)
    Detour = pickle.load(f)
    Validation = pickle.load(f)

# with open('recording_ILPDDQN_200car_1212.pkl', 'rb') as m:
#     loss2 = pickle.load(m)
#     total_reward_plot2 = pickle.load(m)
#     contradiction2 = pickle.load(m)
#     Detour2 = pickle.load(m)
#     Validation2 = pickle.load(m)

# with open('recording_ILPADDQN_200car_further_training_1029.pkl', 'rb') as g:
#     loss3 = pickle.load(g)
#     total_reward_plot3 = pickle.load(g)
#     contradiction3 = pickle.load(g)
#     Detour3 = pickle.load(g)
#     Validation3 = pickle.load(g)
#
# with open('recording_ILPADDQN_200car_further_training_1030.pkl', 'rb') as g:
#     loss4 = pickle.load(g)
#     total_reward_plot4 = pickle.load(g)
#     contradiction4 = pickle.load(g)
#     Detour4 = pickle.load(g)
#     Validation4 = pickle.load(g)
#
# with open('recording_ILPADDQN_200car_further_training_1030_1.pkl', 'rb') as g:
#     loss5 = pickle.load(g)
#     total_reward_plot5 = pickle.load(g)
#     contradiction5 = pickle.load(g)
#     Detour5 = pickle.load(g)
#     Validation5 = pickle.load(g)

# loss.extend(loss2)
# total_reward_plot.extend(total_reward_plot2)
# contradiction.extend(contradiction2)
# Detour.extend(Detour2)
# Validation.extend(Validation2)

# loss.extend(loss3)
# total_reward_plot.extend(total_reward_plot3)
# contradiction.extend(contradiction3)
# Detour.extend(Detour3)
# Validation.extend(Validation3)
#
# loss.extend(loss4)
# total_reward_plot.extend(total_reward_plot4)
# contradiction.extend(contradiction4)
# Detour.extend(Detour4)
# Validation.extend(Validation4)
#
# loss.extend(loss5)
# total_reward_plot.extend(total_reward_plot5)
# contradiction.extend(contradiction5)
# Detour.extend(Detour5)
# Validation.extend(Validation5)


s1 = pd.Series(loss)
rolling_avg1 = s1.rolling(window=10).mean()
rolling_max1 = s1.rolling(window=10).max()
rolling_min1 = s1.rolling(window=10).min()
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