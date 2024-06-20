from data import data_path
from utils.config import end_simulations, ini_simulations
from utils.simulator import FdaSimulator
import os
import pandas as pd
import pickle

# Prepare some global parameters used to generate data
n_basis_simulated_data = 31
sd_x_serie = 0.01
cnt = 30
alpha_p =  1 * cnt
beta_p = 3 * cnt
positions = [0.15, 0.35, 0.55, 0.85]
intercept_brownian = 0
slope_brownian = 1

# data paths
input_data_path = os.path.join(data_path, "input")
output_data_path = os.path.join(data_path, "output")
scenarios_file = "scenarios_braga.csv"

# Explore dataframe that contains the scenarios
df_scenarios = pd.read_csv(os.path.join(input_data_path, scenarios_file))

# Instantiate the class
fda_simulator = FdaSimulator()
times = ["t_" + str(x) for x in fda_simulator.abscissa_points]

datasets_type = ["train", "validation", "test"]
num_simulations = end_simulations - ini_simulations

for _, scenario in df_scenarios.iterrows():
    scenario_id = scenario["scenario_id"]
    type_covariate = scenario["type_covariate"]
    type_transformation = scenario["type_transformation"]
    eta = scenario["eta"]
    sample_size = scenario["sample_size"]
    # Create the folder for the current scenario if it does not exist
    output_dir = os.path.join(output_data_path, f"scenario_{scenario_id}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i_sim in range(num_simulations):
        # Simulate
        X, phi_X, epsilon, beta_data, col_indexes_bct, target = fda_simulator.simulate(
            type_covariate=type_covariate,
            type_transformation=type_transformation,
            sample_size=sample_size,
            eta=eta,
            datasets_type=datasets_type,
            # Series representation
            n_basis_simulated_data=n_basis_simulated_data,
            sd_x=sd_x_serie,
            # Beta parameters
            alpha_param=alpha_p,
            beta_param=beta_p,
            # Brownian parameters
            intercept_brownian=intercept_brownian,
            slope_brownian=slope_brownian,
            positions=positions
        )

        # Store the data
        for i_dataset_type in range(len(datasets_type)):
            dataset_type = datasets_type[i_dataset_type]
            # Transform X and y to pandas objects
            df_X = pd.DataFrame(
                data=X[i_dataset_type],
                columns=times
            )

            df_target = pd.DataFrame(
                data=target[i_dataset_type],
                columns=["target"]
            )
            X_file = os.path.join(output_dir, f"X_sim_{dataset_type}_{i_sim}.csv")
            target_file = os.path.join(output_dir, f"target_sim_{dataset_type}_{i_sim}.csv")
            col_indexes_bct_file = os.path.join(output_dir, f"col_indexes_bct_{dataset_type}_{i_sim}.pkl")
            df_X.to_csv(X_file, index=False)
            df_target.to_csv(target_file, index=False)
            with open(col_indexes_bct_file, 'wb') as f:
                pickle.dump(col_indexes_bct, f)
