from data import data_path
import os
import shutil

data_processed_dir_name = "data_processed"
data_processed_dir = os.path.join(data_path, data_processed_dir_name)
output_dir = os.path.join(data_path, "output")
scenario_names = [x for x in os.listdir(output_dir) if "scenario" in x]
data_dirs = os.listdir(data_path)

if data_processed_dir_name in data_dirs:
    raise FileExistsError(f"{data_processed_dir_name} already exists")
else:
    os.mkdir(data_processed_dir)
ml_models = ["fnn", "knn", "lm"]

for scenario in scenario_names:
    data_output_scenario_dir = os.path.join(output_dir, scenario)
    data_processed_scenario_dir = os.path.join(data_processed_dir, scenario)
    if scenario in os.listdir(data_processed_dir):
        raise FileExistsError(f"{scenario} already exists in {data_processed_dir_name}")
    else:
        os.mkdir(data_processed_scenario_dir)
    shapley_files = [x for x in os.listdir(data_output_scenario_dir) if "shapley_" in x]
    for shapley_file in shapley_files:
        src = os.path.join(data_output_scenario_dir, shapley_file)
        dst = os.path.join(data_processed_scenario_dir, shapley_file)
        shutil.copy(src, dst)
        num_sim = shapley_file[8:-4]
        for ml_model in ml_models:
            r2_file = f"r2_test_{ml_model}_{num_sim}.pkl"
            src = os.path.join(data_output_scenario_dir, r2_file)
            dst = os.path.join(data_processed_scenario_dir, r2_file)
            shutil.copy(src, dst)
