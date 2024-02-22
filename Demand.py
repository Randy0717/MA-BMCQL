import pandas as pd


class Demand_1():
    def __init__(self, demand_path):
        # self.demand = pd.read_csv(demand_path)
        self.demand = pd.read_csv(demand_path)
        self.filtered_demand = self.demand.loc[self.demand['minute'] == 0].reset_index(drop=True)
        self.current_demand = self.demand.loc[self.demand['minute'] == 0].reset_index(drop=True)
        self.episode_time = 0
        self.current_time = 0
        self.num_lost_demand = 0

    def initialization(self, episode_time, p_sample):
        self.filtered_demand = self.demand.sample(frac=p_sample).sort_index()
        mask = (self.filtered_demand['minute'] >= 480) & (self.filtered_demand['minute'] <= 510)
        print("p_sample is:", p_sample)
        print("total number of demand from 8:00 am to 8:30 am at this episode is:", len(self.filtered_demand[mask]))
        self.current_demand = self.filtered_demand.loc[self.filtered_demand['minute'] == episode_time].reset_index(drop=True)
        self.episode_time = episode_time
        self.current_time = episode_time
        self.num_lost_demand = 0

    def update(self):
        self.current_time += 1
        self.current_demand = pd.concat(
            [self.current_demand, self.filtered_demand.loc[self.filtered_demand['minute'] == self.current_time]])
        self.current_demand = self.current_demand.reset_index(drop=True)

        # drop those orders that are not taken over 5 minutes
        if self.current_time >= 5 + self.episode_time:
            self.num_lost_demand += len(self.current_demand[self.current_demand['minute'] <= (self.current_time - 5)])
            self.current_demand = self.current_demand.drop(
                index=self.current_demand[self.current_demand['minute'] <= (self.current_time - 5)].index).reset_index(
                drop=True)

    def pickup(self,unique_r_ids):
        # Convert the set to a list
        unique_r_ids_list = list(unique_r_ids)

        # Drop rows whose index is in unique_r_ids_list
        self.current_demand = self.current_demand.drop(unique_r_ids_list)

        # Reset index
        self.current_demand = self.current_demand.reset_index(drop=True)


