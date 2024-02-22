import matplotlib.pyplot as plt
import numpy as np

# Data: estimated values and actual rewards for each day, along with MPC-greedy values
estimations = [
    [923331, 1085281, 1032330, 1051810, 1075015],  # Estimation 1
    [845578, 878704, 876260, 886573, 888965],      # Estimation 2
    [1619506, 1771390, 1788337, 1772256, 1719442]  # Estimation 3
]

rewards = [
    [773159, 676476, 653554, 597715, 729747],  # Reward 1
    [789633, 821811, 860585, 838730, 852531],  # Reward 2
    [673016, 700090, 714827, 737806, 744233]   # Reward 3
]

# Define positions for each bar chart
days = np.arange(len(estimations[0]))  # Days of the week

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Width of the bar chart
bar_width = 0.25  # Adjusted width to fit three bars

# Colors and opacity settings
reward_alpha = 1.0  # Opacity for the reward bars
excess_alpha = 0.5  # Opacity for the excess bars

# Function to plot bars
def plot_bars(position, rewards, estimations, label, color):
    ax.bar(days + position * bar_width, rewards, bar_width,
           alpha=reward_alpha, label=label, color=color)
    estimation_excess = np.array(estimations) - np.array(rewards)
    ax.bar(days + position * bar_width, estimation_excess, bar_width,
           alpha=excess_alpha, bottom=rewards, color=color)

# Plotting the bars for each set of rewards and estimations
plot_bars(0, rewards[0], estimations[0], 'ILPDDQN', 'blue')
plot_bars(1, rewards[1], estimations[1], 'MA-BMCQL', 'purple')
plot_bars(2, rewards[2], estimations[2], 'ILPAC', 'brown')

# Add some text labels
ax.set_xlabel('Day of the Week')
ax.set_ylabel('Values')
ax.set_title('Estimation vs. Reward for Different Frameworks')
ax.set_xticks(days + bar_width)
ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])

# Add legend
ax.legend()

# Show plot
plt.show()