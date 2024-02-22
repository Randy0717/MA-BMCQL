from Demand import Demand_1
from Vehicles_ILPDDQN import Vehicles
from Central_ILPDDQN import Central
from utility import time_record, training_curve, training_record
import time
import pickle
import random


EPS_START = time.time()
# random.seed(0)

# Hyperparameter settings
# DEMAND_PATH = '..\csv\demand_new.csv'
EPISODE_TIME = 480
TEST_TIME = 30
TEST_VEHICLE = 0
NUM_VEHICLES = 1000
NUM_EPISODES = 200
EPSILON = 0.1
exploration_rate = EPSILON
EPSILON_FINAL = 0.1
EPSILON_DECAY_Rate = 0.996
validation_eps = 201

# SAVE PATH
Load_Path = 'Save/DQN_CP+TR_ILPDDQN_1000car_1227_WED_100percent_005tau.pt'
# POLICY_PATH = 'Save/DQN_CP+TR_ILPDDQN_1000car_1227_WED_100percent_005tau.pt'
# training_curve_path = 'Training plot/training_ILPDDQN_1000car_1227_WED_100percent_005tau.png'
# training_record_path = 'recording/recording_ILPDDQN_1000car_1227_WED_100percent_005tau.pkl'

# Simulation System Initialization
Initial_start = time.time()
# DEMAND INITIALIZATION
# Demand = Demand(DEMAND_PATH)
# Demand.initialization(EPISODE_TIME)
# VEHICLE INITIALIZATION
Vehicles = Vehicles(NUM_VEHICLES)
Vehicles.load(load_path=Load_Path)
# CENTRAL INITIALIZATION
Central = Central()
Initial_end = time.time()
print("Initialization takes {0} seconds".format(Initial_end-Initial_start))

# evaluation benchmarks
contradiction_rate_eps = []
total_reward_eps = []
average_detour_eps = []
training_loss_steps = []
estimation_eps = []
validation_reward_eps = []


for eps in range(NUM_EPISODES+1):
    print("\n -------- episode {0} starts --------".format(eps+1))
    if eps % validation_eps == 0:
        EPISODE_TIME = 480
    else:
        EPISODE_TIME = 480

    temp = 4
    DEMAND_PATH = '..\csv_2023_1122\demand_new_Day'+ str(temp) + '.csv'
    Demand = Demand_1(DEMAND_PATH)
    print('In episode {0} today is Day {1}'.format(eps, temp))
    p_sample = 1
    Demand.initialization(EPISODE_TIME, p_sample)
    Central.reset()
    Vehicles.reset(NUM_VEHICLES)
    exploration_rate = max(exploration_rate * EPSILON_DECAY_Rate, EPSILON_FINAL)
    Estimation = 0
    still_idle_cars = [i for i in range(NUM_VEHICLES)]

    for delta_t in range(TEST_TIME):
        Demand.update()

        print("\ncurrent time is {0}, exploration rate is {1} ".format(Demand.current_time,exploration_rate))
        print("num of Demands is {0}".format(len(Demand.current_demand)))
        start = time.time()

        # Get Action and Value matrix
        if eps % validation_eps == 0:
            # validation
            Vehicles.observe(Demand.current_demand, Demand.current_time - (EPISODE_TIME - 480), exploration_rate= 0)
        else:
            # training
            Vehicles.observe(Demand.current_demand, Demand.current_time - (EPISODE_TIME - 480), exploration_rate= exploration_rate)

        end1 = time.time()
        print('vehicle observe takes {0}'. format(end1 - start))

        # Central solve ILP and get assignment
        print('assignment starts')
        assign_time = time.time()
        Assignment_table, Obj = Central.assign2(Vehicles.Value_Matrix)

        # assign_time_end = time.time()
        # print('ILP solving takes', assign_time_end-assign_time)

        # if delta_t == 0:
        #     Estimation = Obj
        #     print(Estimation)

        i = TEST_VEHICLE
        if Assignment_table[i] is not None:
            order_index = Assignment_table[i]
            print('Vehicle {0} assigned state is {1}'.format(i, Vehicles.S[i][order_index]))
            print('Vehicle {0} assigned order is {1}'.format(i, Vehicles.S[i][order_index][11:13]))
            print('Vehicle {0} assigned action is {1}'.format(i, Vehicles.Action_Matrix[i][order_index]))
            print('Vehicle {0} assigned q value is {1}'.format(i, Vehicles.Value_Matrix[i][order_index]))

        for index in still_idle_cars:
            if Assignment_table[index] is not None:
                order_index = Assignment_table[index]
                Estimation += Vehicles.Value_Matrix[index][order_index]
                still_idle_cars.remove(index)

        feedback_table, x_table, new_route_table, new_route_time_table\
            = Central.feedback(Vehicles.S, Vehicles.Action_Matrix, Assignment_table, Demand.current_demand,
                               Vehicles.zone_lookup)
        # end4 = time.time()

        #
        if feedback_table[TEST_VEHICLE] is not None:
            print(" vehicle {0}'s reward is {1} ".format(TEST_VEHICLE, feedback_table[TEST_VEHICLE][2]))
        #
        Vehicles.update(feedback_table, x_table, new_route_table,\
                        new_route_time_table, Demand.current_time, EPISODE_TIME, TEST_TIME)
        # end5 = time.time()

        print("vehicle {0}'s updated information is {1}".format(TEST_VEHICLE, Vehicles.X[TEST_VEHICLE]))
        # # Vehicles.draw_map(Demand.current_time, TEST_VEHICLE)
        # if eps % validation_eps != 0 and delta_t % 2 == 0:
        #     Vehicles.learn()

        # 使用filter函数移除None元素
        filtered_assignment = []
        for i in range(len(Assignment_table)):
            if Assignment_table[i] is not None:
                if Vehicles.Action_Matrix[i][Assignment_table[i]] >0:
                    filtered_assignment.append(Assignment_table[i])
        Demand.pickup(filtered_assignment)
        Central.Pickup += len(filtered_assignment)
        # end6 = time.time()
        #
        # #time record of this minute
        # # time_record(start, end1, end2, end3, end4, end5, end6)


    average_detour = sum(Central.Total_Detour)/len(Central.Total_Detour)
    if eps % validation_eps != 0:
        total_reward_eps.append(Central.Total_Reward)
        average_detour_eps.append(average_detour)
        training_loss_steps = Vehicles.training_steps

    else:
        print('\n validation episode: ', eps+1)
        print('No. orders being picked up:', Central.Pickup)
        print("Estimation is:", Estimation)
        estimation_eps.append(Estimation)
        print('Total reward is:', Central.Total_Reward)
        print('Average detour:', average_detour)
        validation_reward_eps.append(Central.Total_Reward)

    if len(Vehicles.M) == 50000:
        transition_tuples = Vehicles.M

        # 将数据序列化并保存到文件
        with open('Transitions/transitions.pkl', 'wb') as f:
            pickle.dump(transition_tuples, f)
        break

EPS_END = time.time()
print("\n whole transition sampling takes {0} hours".format(int(EPS_END-EPS_START)/3600))
