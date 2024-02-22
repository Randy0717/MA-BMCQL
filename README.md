# A.Demand Preparations
The trip requests data has already been prepared for you in csv/demand_new.csv, to run this program, u need to unzip csv directory for demand_new.csv and zone_table.csv. Moreover, you also need to download csv_2023_1122 directory from the given link in csv_2023_1122 file for whole month order requests.

The environment for this study is based on the public dataset of taxi trips provided by Manhattan, New York City. We extracted 30 minutes' worth of trip order requirements during the peak hours (specifically from 8:00 am to 10:00 am) in May 2016, and divided Manhattan into 57 zones. This zoning was informed by the distribution of orders and a resolution of 800m x 800m was used. The visualization is shown in Figure Demand Visualization.png

# B.OSRM Implementation 
The route guidance, driving route time estimation, driving route update, and driving route visualization in the simulation model are provide by osrm_router.py and Dijkstra.py, by solving Travel Sallings Man problem.

However, to implement and test the codes, you have to follow the official gudiance of OSRM to use docker to create local server. The links are provided in our paper.

# C. Transitions
For offline training comparisons, we pre-collect the vehicle transitions by running greedy ride-pooling policy with 20% chance to explore (which is labeled as medium in our paper) for Mondays to Fridays, we also ran the online trained ILPDDQN policy with 10 percent chance to explore to get a 'good' transition dataset for Wednesday. 

Further details of the six transition datasets could be found below file path: 'ILPCQL_20240120_CP_offline\Transitions\XXX'

# D. MA-BMCQL Structure
The architecture of the MA-BMCQL Framework can be better understood through Figure below.

![MA-BMCQL Framework](/Overview_of_MA-BMCQL_Frameworks.png "Overview of MA-BMCQL Framework")

You may check the details by reading through the lines or our paper draft.

# E. Training Simulation Framework
After preparing the stuffs above, you may run the simulation framework in simulator_xxx.py under each file like 'ILPCQL_20240120_CP_offline' for example, which use parallel computing and will save neural network parameters in Save directory, and save training plot in Training plot directory during the training process. 

Also you may read the past training record by running read.py in the recording file. Moreover, you may change some hyperparameters and the save path. Some of our training parameters and networks are also given.

However, before running, you need to first keep your local OSRM server activated. Also, you may need to change the input file path according to your own computer's settings.

# F. Validation & Benchmarks
You may also directly check the validation results by referring to the plots and figures under file 'Comparisons'. Our comprehensive experiments and validations across different weekdays reveal that MA-BMCQL dramatically accelerates policy training—reducing time requirements by over 20-fold—while also delivering superior performance compared to benchmark reinforcement learning frameworks like ILPAC and ILPDDQN in ride-sharing under all tested offline training scenarios by cutting down the tremendous overestimation due to out of distribution data sampling and multi-agent interactions.

You may also load some of our pretrained network parameters under 'Save'. The codes will print out the total rewards at the end of the episode.

Still, before running, you need to first keep your local OSRM server activated. Also, you may need to change the input file path according to your own computer's settings.

# Final Warning Note:
Please quote our codes or paper 'MA-BMCQL: On Efficient Simulator-free Ride-Sharing Dispatch Training with Multi-Agent Bipartite Match Conservative Q-Learning' if u use for publication or commercial purposes.

Thanks.
