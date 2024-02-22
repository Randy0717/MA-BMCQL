from osrm_router import TSP_route
from transit import generate_transit_edge, ETA_Transit
from utility import geodistance, zone_order_map
from joblib import Parallel, delayed
from pulp import PULP_CBC_CMD, LpProblem, LpMaximize, LpVariable, lpSum
import pulp
from scipy.optimize import linear_sum_assignment
import ast
import numpy as np

mode = 1
if mode == 1:
    action_per_order = 57
else:
    action_per_order = 1

beta0 = 100 # 30
beta1 = 40
beta2 = 5
beta3 = 2 # 2 10
beta4 = 20 # 10
threshold = 15 #15 10
# penalty = 20
discount_factor = 0.99

class Central():
    def __init__(self, mode='CP+TR'):
        G, Stations = generate_transit_edge(200)
        self.Transit_G = G
        self.Transit_Stations = Stations
        self.Contradiction = 0
        self.Pickup = 0
        self.Total_Reward = 0
        self.Total_Detour = []
        self.mode = mode

    def reset(self):
        self.Contradiction = 0
        self.Pickup = 0
        self.Total_Reward = 0
        self.Total_Detour = []

    def Transit(self, Olat, Olon, Dlat, Dlon):
        time, path = ETA_Transit(Olat, Olon, Dlat, Dlon, self.Transit_G, self.Transit_Stations)

        return time, path

    def OSRM(self, origin_point, destination_points):
        route, route_t, t = TSP_route(origin_point, destination_points)[:-1]

        return route, route_t, t

    def assign1(self, Value_Matrix):
        # 创建一个新的模型
        model = LpProblem("ilp", LpMaximize)

        num_vehicles = len(Value_Matrix)
        num_demand = len(Value_Matrix[0])

        # 创建变量
        x = [[LpVariable(f"x_{i}_{j}", cat='Binary') for j in range(num_demand)] for i in range(num_vehicles)]

        # 设置目标
        model += lpSum(Value_Matrix[i][j] * x[i][j] for i in range(num_vehicles) for j in range(num_demand))

        # 添加约束
        for i in range(num_vehicles):
            model += lpSum(x[i][j] for j in range(num_demand)) <= 1  # 每行只能选择一个

        for j in range(num_demand):
            model += lpSum(x[i][j] for i in range(num_vehicles)) <= 1  # 每列只能选择一个

        # 求解模型
        model.solve(PULP_CBC_CMD(msg=False))

        # 创建一个列表来保存每个车辆被分配的订单
        assignment = [None] * num_vehicles

        # 获取每个车辆被分配的订单
        for i in range(num_vehicles):
            for j in range(num_demand):
                if x[i][j].varValue > 0.5:  # 如果x[i,j]等于1
                    assignment[i] = j  # 车辆i被分配订单j

        # 返回分配结果
        return assignment, pulp.value(model.objective)

    def assign2(self, Value_Matrix):
        num_vehicles = len(Value_Matrix)
        num_demands = len(Value_Matrix[0])
        Value_Matrix = np.column_stack((Value_Matrix, np.zeros((num_vehicles, num_vehicles))))

        # 首先，我们需要将最大化问题转换为最小化问题，
        # 因为 linear_sum_assignment 默认解决的是最小化问题。
        # 通过将所有的值乘以-1，我们可以轻松地实现这一点。
        cost_matrix = -1 * np.array(Value_Matrix)

        # 使用 linear_sum_assignment 函数来找到最优分配
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        # print(row_indices, col_indices)

        # 创建一个列表来保存每个车辆被分配的订单
        assignment = [None] * len(Value_Matrix)

        # 获取每个车辆被分配的订单
        for i in range(len(row_indices)):
            if col_indices[i] >= num_demands:
                assignment[row_indices[i]] = None
            else:
                assignment[row_indices[i]] = col_indices[i]

        # 计算最大化的值
        max_value = -1 * cost_matrix[row_indices, col_indices].sum()

        # 返回分配结果和最大值
        return assignment, max_value

    def feedback(self, states, actions, Assignment, demand_current, zone_lookup):
        feedback_table = []
        x_table = []
        new_route_table = []
        new_route_time_table = []

        results = Parallel(n_jobs=16)(
            delayed(excute)(states, actions, assignment, demand_current, zone_lookup, self.OSRM, self.Transit, geodistance,
                            self.mode)
            for states, actions, assignment in zip(states, actions, Assignment))

        # results = []
        # for decision, assignment in zip(decision_table, Assignment_table):
        #     results.append(excute(decision, assignment, demand_current, zone_lookup, self.OSRM, self.Transit, geodistance))

        for i in range(len(results)):
            result = results[i]
            feedback_table.append(result[0])
            x_table.append(result[1])
            new_route_table.append(result[2])
            new_route_time_table.append(result[3])
            if Assignment[i] is not None:
                self.Total_Reward += discount_factor ** (result[0][0][-1]) * result[0][2]
                self.Total_Detour.append(result[1][7 + 3-int(result[1][1])])

        return feedback_table, x_table, new_route_table, new_route_time_table

def excute(states, actions, assignment, demand_current, zone_lookup, TSP_route, ETA_Transit, geodistance, mode):
    # output reward , new_route, new_route_time
    if states is not None and assignment is not None:
        s = states[assignment]
        a = actions[assignment]

        if a == 0:
            reward = 0
            x = s[:11]
            x.append(0)
            x.append(0)
            new_route = None
            new_route_time = None
            feedback = [s, a, reward, 1]

        else:
            x_loc = ast.literal_eval(zone_lookup.loc[s[0], '(lat,lon)'])
            occupied_seats = int(3 - s[1])
            r_id = assignment
            contradiction = 0
            plat, plon, dlat, dlon = demand_current.loc[r_id, 'plat'], demand_current.loc[r_id, 'plon'], \
                demand_current.loc[r_id, 'dlat'], demand_current.loc[r_id, 'dlon']
            pzone, dzone = demand_current.loc[r_id, 'pzone'], demand_current.loc[r_id, 'dzone']
            # print(pzone, dzone)
            # zone = dzone
            # if mode == "CP+TR":
            #     oloc = ast.literal_eval(zone_lookup.loc[zone, '(lat,lon)'])
            #     # print(oloc)
            # elif mode == "CP":
            #     oloc = (dlat, dlon)
            oloc = (dlat, dlon)

            # 0. direct_distance
            direct_distance = geodistance(plat, plon, dlat, dlon)
            direct_time = int((direct_distance*1.3/40)*60)
            # print('direct time is:', direct_time)

            # 1. pickup
            # pickup_route, pickup_route_t, pickup_time = TSP_route(x_loc, [(plat, plon)])
            pickup_time = [int((geodistance(x_loc[1],x_loc[0],plon,plat)*1.3/40)*60)]

            # 2. carpooling
            destination_points = []
            for i in range(occupied_seats):
                onboard_loc = ast.literal_eval(zone_lookup.loc[s[5+i], '(lat,lon)'])
                destination_points.append(onboard_loc)
            destination_points.append(oloc)
            new_route, new_route_time, new_time = TSP_route((plat, plon), destination_points)
            # print('travel time is', new_time)

            if len(new_route) == 0:
                a = 0
                reward = 0
                x = s[:11]
                x.append(0)
                x.append(0)
                feedback = [s, a, reward, 1]
                new_route = None
                new_route_time = None

            else:
                # 3. Transit
                # if mode == "CP+TR":
                #     transit, path = ETA_Transit(oloc[0], oloc[1], dlat, dlon)
                #     if a == dzone:
                #         transit = 0
                #
                # elif mode == "CP":
                #     transit = 0
                transit = 0

                # 4. direct transfer
                # direct_route, direct_route_time, direct_time = TSP_route((plat, plon), [destination_points[-1]])
                # direct_time = direct_time[0]

                # 5. reward calculation
                original_total_travel_time = sum(s[2:2+occupied_seats]) + direct_time
                total_travel_time = sum(new_time) + transit + pickup_time[0] * occupied_seats

                add_time = total_travel_time - original_total_travel_time

                # print('direct ditance is:', direct_distance)
                # print('pickup time is:', pickup_time)
                # print('detour time is:', add_time)

                if add_time > threshold:
                    reward = beta0 + beta1 * direct_distance - beta2 * pickup_time[0] - beta3 * threshold - beta4 * (
                                add_time - threshold)
                else:
                    reward = beta0 + beta1 * direct_distance - beta2 * pickup_time[0] - beta3 * add_time

                x = s[:11]
                if contradiction == 1:
                    reward = 0
                    x.append(0)
                    x.append(0)
                    new_route = None
                    new_route_time = None
                    pickup_time = [0]

                elif contradiction == 0 and reward <= -30:
                    reward = -30
                    x.append(0)
                    x.append(0)
                    new_route = None
                    new_route_time = None
                    pickup_time = [0]

                else:
                    # reward = beta0 + beta1 * direct_distance
                    x[1] -= 1
                    x[0] = s[11]
                    x[8: 8 + occupied_seats] = list(np.array(x[8: 8 + occupied_seats]) + np.array(new_time[:-1]) - np.array(x[2:2 + occupied_seats]))
                    x[8 + occupied_seats] = transit + new_time[-1] - direct_time
                    x[2: 2 + occupied_seats + 1] = new_time
                    x[5 + occupied_seats] = dzone
                    x.append(1)
                    x.append(pickup_time[0] + 1)

                feedback = [s, a, reward, pickup_time[0]+1]

    else:
        feedback = None
        new_route = None
        new_route_time = None
        x = None

    return feedback, x, new_route, new_route_time












