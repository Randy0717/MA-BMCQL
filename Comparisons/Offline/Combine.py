import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open('recording_ILPDDQN_1000car_0124_WED_100percent_offline+cql400.pkl', 'rb') as f3:
    loss1 = pickle.load(f3)
    total_reward_plot1 = pickle.load(f3)
    contradiction1 = pickle.load(f3)
    Detour1 = pickle.load(f3)
    Validation1 = pickle.load(f3)

with open('recording_ILPDDQN_1000car_0124_WED_100percent_offline+cql400_fur.pkl', 'rb') as f1:
    loss2 = pickle.load(f1)
    total_reward_plot2 = pickle.load(f1)
    contradiction2 = pickle.load(f1)
    Detour2 = pickle.load(f1)
    Validation2 = pickle.load(f1)

loss1.extend(loss2)
total_reward_plot1.extend(total_reward_plot2)
contradiction1.extend(contradiction2)
Detour1.extend(Detour2)
Validation1.extend(Validation2)

combined_loss = loss1
combined_total_reward_plot = total_reward_plot1
combined_contradiction = contradiction1
combined_Detour = Detour1
combined_Validation = Validation1

# Specify the name of the new output file
output_filename = 'recording_ILPDDQN_1000car_0124_WED_100percent_offline_cql400.pkl'

# Open the file in binary write mode and dump the combined data
with open(output_filename, 'wb') as f_out:
    pickle.dump(combined_loss, f_out)
    pickle.dump(combined_total_reward_plot, f_out)
    pickle.dump(combined_contradiction, f_out)
    pickle.dump(combined_Detour, f_out)
    pickle.dump(combined_Validation, f_out)

print(f"Data successfully saved to {output_filename}")