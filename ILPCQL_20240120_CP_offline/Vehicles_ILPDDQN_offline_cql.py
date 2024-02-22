import pandas as pd
import numpy as np
import pickle
import ast
import folium
import torch
import torch.nn as nn

from joblib import Parallel, delayed
from osrm_router import update_loc, get_map
from utility import geodistance
from CQL import *

with open('Transitions/transitions_75k_wed.pkl', 'rb') as f:
    loaded_tuples = pickle.load(f)

if torch.cuda.is_available():
    device = torch.device("cuda")          # Use GPU

else:
    device = torch.device("cpu")           # Use CPU
    print('PyTorch is using CPU.')

VACANT_SEATS = 3
Capacity = len(loaded_tuples)
df = pd.read_csv('..\csv\demand_new.csv')
print(Capacity)

class Vehicles():
    def __init__(self, num_vehicles):
        ## Vehicle information initialization
        # X: num_vehicles by 13 size, where only the previous 11 is what we will extract for its own state
        # 0 --- current zones
        # 1 --- vacant seats
        # 2-4 --- remaining travel time on car of each passenger
        # 5-7 --- drop-off destinations of each passenger
        # 8-10 --- total additional time occurred due to pooling+transit of each passenger
        # 11 --- Vehicle state 0: available 1: picking up 2: full
        # 12 --- remaining pickup time

        self.zone_lookup = pd.read_csv('..\csv\zone_table.csv')

        X = np.zeros((num_vehicles, 13))

        # Initialize Vehicles' zone information and vacant seats
        random_integers = np.random.randint(0, len(self.zone_lookup), size=(num_vehicles, 1))
        X[:, 0] = df[:num_vehicles]['pzone']
        X[:, 1] = VACANT_SEATS

        # Vehicles' route and route time recorder X_travel & X_travel_time
        X_travel_route = []
        X_travel_time = []
        X_experience = []
        X_obsorders = []
        # X_gps = []

        Value_Matrix = []
        Action_Matrix = []

        S = []
        # each passenger's real destinations on board
        X_real_dest = np.zeros((num_vehicles, VACANT_SEATS * 2))

        for i in range(num_vehicles):
            X_travel_route.append([])
            X_travel_time.append([])
            X_experience.append([])
            X_obsorders.append([])
            S.append([])
            Value_Matrix.append([])
            Action_Matrix.append([])
            # loc = ast.literal_eval(self.zone_lookup.loc[X[i,0], '(lat,lon)'])
            # X_gps.append([loc[0],loc[1]])

        self.number = num_vehicles
        self.zone_lookup = self.zone_lookup
        self.X = X
        self.X_travel_route = X_travel_route
        self.X_travel_time = X_travel_time
        self.X_real_dest = X_real_dest
        self.X_experience = X_experience
        self.X_observe_orders = X_obsorders
        self.S = S
        self.Value_Matrix = Value_Matrix
        self.Action_Matrix = Action_Matrix

        # Policy Parameter
        self.DQN_target = DQN(dinput=14, doutput=1 + 1).to(device)
        self.DQN_training = DQN(dinput=14, doutput=1 + 1).to(device)
        self.optimizer = optim.Adam(self.DQN_training.parameters(), lr= 0.01) # 0.001 0.01 0.005
        self.criterion = nn.MSELoss()
        self.num_of_train_steps = 0
        self.training_steps = []
        self.M = loaded_tuples

    def reset(self, num_vehicles):
        X = np.zeros((num_vehicles, 13))

        # Initialize Vehicles' zone information and vacant seats
        random_integers = np.random.randint(0, len(self.zone_lookup), size=(num_vehicles, 1))
        X[:, 0] = df[:num_vehicles]['pzone']
        X[:, 1] = VACANT_SEATS

        # Vehicles' route and route time recorder X_travel & X_travel_time
        X_travel_route = []
        X_travel_time = []
        X_experience = []
        X_obsorders = []
        # X_gps = []
        Value_Matrix = []
        Action_Matrix = []
        S = []
        # each passenger's real destinations on board
        X_real_dest = np.zeros((num_vehicles, VACANT_SEATS * 2))

        for i in range(num_vehicles):
            X_travel_route.append([])
            X_travel_time.append([])
            X_experience.append([])
            X_obsorders.append([])
            Value_Matrix.append([])
            Action_Matrix.append([])
            S.append([])
            # loc = ast.literal_eval(self.zone_lookup.loc[X[i,0], '(lat,lon)'])
            # X_gps.append([loc[0],loc[1]])

        self.number = num_vehicles
        self.zone_lookup = self.zone_lookup
        self.X = X
        self.X_travel_route = X_travel_route
        self.X_travel_time = X_travel_time
        self.X_real_dest = X_real_dest
        self.X_experience = X_experience
        self.X_observe_orders = X_obsorders
        self.S = S
        self.Value_Matrix = Value_Matrix
        self.Action_Matrix = Action_Matrix

    def observe(self, demand_current, current_time, exploration_rate):
        if self.number >= 100:
            results = Parallel(n_jobs=16, backend='loky')(
                delayed(vehicle_observe)(x, self.zone_lookup, demand_current, geodistance, current_time,
                                         self.DQN_training, exploration_rate)
                for x in self.X)
        else:
            results = Parallel(n_jobs=-1, backend='threading')(
                delayed(vehicle_observe)(x, self.zone_lookup, demand_current, geodistance, current_time,
                                         self.DQN_training, exploration_rate)
                for x in self.X)

        self.S = []
        self.Action_Matrix = []
        self.Value_Matrix = []
        for result in results:
            self.S.append(result[0])
            self.Action_Matrix.append(result[1])
            self.Value_Matrix.append(result[2])
    def decide(self, exploration_rate=0):
        # decision_table = Parallel(n_jobs=-1)(
        #         delayed(vehicle_decide)(s,self.DQN_training, exploration_rate) for s in self.S)
        decision_table = []
        for s in self.S:
            decision_table.append(vehicle_decide(s, self.DQN_training, exploration_rate))

        return decision_table

    def update(self, feedback_table, x_table, new_route_table, new_route_time_table,
               current_time, episode_time, test_time):
        for i in range(self.number):
            feedback = feedback_table[i]
            x = x_table[i]
            new_route = new_route_table[i]
            new_route_time = new_route_time_table[i]

            # if feedback is not None:
            #     # experience load and put into buffer
            #     if len(self.X_experience[i]) > 0:
            #         self.X_experience[i].append(feedback[0][-1] - self.X_experience[i][0][-1])
            #         self.X_experience[i].append(feedback[0])
            #
            #     else:
            #         self.X_experience[i].append(feedback[0])
            #
            #     if len(self.X_experience[i]) == 5:
            #         self.M = buffer(self.X_experience[i], self.M)
            #         self.X_experience[i] = []
            #         self.X_experience[i].append(feedback[0])
            #
            #     self.X_experience[i].append(feedback[1])
            #     self.X_experience[i].append(feedback[2])
            #
            #     if current_time == episode_time+test_time:
            #         self.X_experience[i].append(30 - self.X_experience[i][0][-1])
            #         self.X_experience[i].append(None)
            #         if len(self.X_experience[i]) == 5:
            #             self.M = buffer(self.X_experience[i], self.M)
            #             self.X_experience[i] = [feedback[0]]
            #
            if feedback is None or feedback[1] == 0 or new_route is None:
                continue

            # next time information, new_route, new_route_time loading from feedback
            self.X[i] = x
            self.X_travel_route[i] = new_route
            self.X_travel_time[i] = new_route_time

        for n in range(self.number):
            x = self.X[n]
            occupied_seats = 3 - int(x[1])
            # if still picking up
            if x[11] == 1:
                if x[12] >= 1:
                    x[12] = x[12] - 1
                    # if remaining picking up time is 0:
                    if x[12] <= 0:
                        # check whether is full
                        if x[1] == 0:
                            x[11] = 2
                        else:
                            x[11] = 0

            else:
                if x[1] < 3:  # if with passenger, update its next location
                    loc, route, route_t = update_loc(60, self.X_travel_route[n], self.X_travel_time[n])
                    zone_distance = []
                    for j in range(len(self.zone_lookup)):
                        zone_loc = ast.literal_eval(self.zone_lookup.loc[j, '(lat,lon)'])
                        dist = geodistance(loc[1], loc[0], zone_loc[1], zone_loc[0])
                        zone_distance.append(dist)
                    # sorted() to sort the distance
                    smallest = sorted(enumerate(zone_distance), key=lambda x: x[1])[:1]
                    smallest_index = [i for i, _ in smallest]
                    # print(smallest_index)
                    x[0] = smallest_index[0]

                    self.X_travel_route[n] = route
                    self.X_travel_time[n] = route_t
                    x[2: 2 + occupied_seats] -= 1

                    # if passenger arrives at its drop-off place:
                    x = dropoff(x)

            self.X[n] = x

    def draw_map(self, current_time, TEST_VEHICLE, time_interval=3):
        if current_time % time_interval == 0:
            # get information
            route = self.X_travel_route[TEST_VEHICLE]
            x = self.X[TEST_VEHICLE]
            loc = ast.literal_eval(self.zone_lookup.loc[x[0], '(lat,lon)'])
            occupied_seats = 3 - int(x[1])
            destination_points = []
            if occupied_seats > 0:
                for i in range(occupied_seats):
                    destination_points.append(ast.literal_eval(self.zone_lookup.loc[x[5 + i], '(lat,lon)']))
            # draw the map
            map = folium.Map(location=[40.81179592602443, -73.96498583811469], zoom_start=13)
            map = get_map(route, loc, destination_points, map)
            map.save("Validation/CP+TR_Parallel/" + "_" + str(current_time) + ".html")

    def learn(self, CAPACITY=Capacity):
        if len(self.M) == CAPACITY:
            # training_temp = 1 if exploration_rate > EPSILON_FINAL else 5
            training_temp = 1
            for m in range(training_temp):
                self.num_of_train_steps += 1
                # print("\n------training steps {0} ------\n".format(self.num_of_train_steps))
                # print("----episode", k, "training at time", t, "starts----")
                train(self.M, self.DQN_target, self.DQN_training, self.optimizer, self.criterion)
                training_loss = calc_loss(self.M, self.DQN_target, self.DQN_training, self.criterion).item()
                # print("\ntraining loss is\n", training_loss)
                self.training_steps.append(training_loss)
                # if self.num_of_train_steps % 2000 == 0:
                #     self.DQN_target.load_state_dict(self.DQN_training.state_dict())
            tau = 0.005  # 软更新的参数τ，可以根据需要进行调整 0.001 0.01 0.005
            for target_param, train_param in zip(self.DQN_target.parameters(), self.DQN_training.parameters()):
                target_param.data.copy_(tau * train_param.data + (1.0 - tau) * target_param.data)

    def save(self, policy_path):
        torch.save(self.DQN_training.state_dict(), policy_path)

    def load(self, load_path):
        self.DQN_target.load_state_dict(torch.load(load_path))
        self.DQN_training.load_state_dict(torch.load(load_path))


def vehicle_observe(x, zone_lookup, demand_current, geodistance_function, current_time, DQN_training, exploration_rate):
    num_demand = len(demand_current)
    if x[11] == 0:
        loc = ast.literal_eval(zone_lookup.loc[x[0], '(lat,lon)'])
        State_matrix = []
        kept_indices = []
        Value_Matrix = -10000 * np.ones(num_demand)
        Action_Matrix = np.zeros(num_demand)
        for j in range(num_demand):
            dist = geodistance_function(loc[1], loc[0], demand_current.loc[j, 'plon'],
                                        demand_current.loc[j, 'plat'])
            s = list(x)[:11]
            s.extend([float(demand_current.loc[j, 'pzone']),
                      demand_current.loc[j, 'dzone']])
            s.append(current_time - 480)
            if dist <= 1.2:
                kept_indices.append(j)

            State_matrix.append(s)

        if kept_indices:  # 如果kept_indices不为空
            # print(len(kept_indices))
            Action_Matrix_temp, Value_Matrix_temp = vehicle_decide(np.array([State_matrix[i] for i in kept_indices]),
                                                                   DQN_training, exploration_rate)
            for idx, a, v in zip(kept_indices, Action_Matrix_temp, Value_Matrix_temp):
                Action_Matrix[idx] = a
                Value_Matrix[idx] = v

        State_matrix = np.array(State_matrix).tolist()

    else:
        State_matrix = None
        Value_Matrix = -100000 * np.ones(num_demand)
        Action_Matrix = np.zeros(num_demand)

    return State_matrix, Action_Matrix, Value_Matrix

def vehicle_decide(state_matrix, DQN_training, exploration_rate):
    num_states = state_matrix.shape[0]
    state_matrix = torch.tensor(state_matrix, dtype=torch.float32).to(device)

    if random.random() > exploration_rate:
        actions = DQN_training(state_matrix).argmax(1)
        q_values = DQN_training(state_matrix).gather(1, actions.unsqueeze(1)).squeeze(1).detach()

    else:
        # choices = np.array(list(range(0, 1)))

        # 使用np.random.choice()函数选择actions
        # actions_np = np.array([
        #     np.random.choice(choices, p=get_probabilities(state_matrix[i, -2].item()))
        #     for i in range(num_states)
        # ])

        values_np = 100000 * np.ones(num_states)

        # 转换成torch tensor
        actions = torch.tensor(np.random.randint(0, 2, size=num_states), dtype=torch.int64, device=device)
        # q_values = DQN_training(state_matrix).gather(1, actions.unsqueeze(1)).squeeze(1).detach()
        q_values = torch.tensor(values_np)

    return actions, q_values

def get_probabilities(penultimate_element): #%75概率CP+TR %25概率CP
    probabilities = np.array([1/58]*58) # 0.75
    probabilities[int(penultimate_element)] += 0 # 0.25
    return probabilities

# 3 vehicle drop_off:
def dropoff(x):
    occupied_seats = 3 - int(x[1])
    passenger = []
    get_off = []

    for i in range(occupied_seats):
        passenger.append(i)
        if x[2 + i] <= 0:
            get_off.append(i)
            if x[11] == 2:
                x[11] = 0
    #
    # print(passenger)
    # print(get_off)
    x[1] += len(get_off)
    for i in range(len(get_off)):
        passenger.remove(get_off[i])

    # print(passenger)
    occupied_seats = 3 - int(x[1])

    temp = np.zeros(11)
    for i in range(occupied_seats):
        temp[2 + i] = x[2 + passenger[i]]
        temp[5 + i] = x[5 + passenger[i]]
        temp[8 + i] = x[8 + passenger[i]]

    x[2:11] = temp[2:]
    return x


# 4 vehicle puts his current experience into replay buffer:
def buffer(X_experience, M, CAPACITY=Capacity):
    if len(M) < CAPACITY:
        M.append(X_experience)
    else:
        M.remove(M[0])
        M.append(X_experience)

    return M
