from utils.logger import logger
from data import data_path
from hyperopt.fnn import HyperOptFnn
from hyperopt.sklearn_gridsearch import HyperOptScikitFda
from shapley.shapley_fda import ShapleyFda
from skfda.ml.regression import KNeighborsRegressor, LinearRegression
from skfda.representation.basis import BSplineBasis
from skfda.representation.grid import FDataGrid
from utils.config import end_simulations, ini_simulations
from utils.utils_workflow import get_abscissa_points, obtain_score, predict_no_verbose, predict_np_reshape, read_data
import keras_tuner
import numpy as np
import os
import pickle

num_intervals = 20
num_permutations = 1000
max_trials_fnn = 10
domain_range = (0, 1)
get_lm_results = True
get_knn_results = True
get_fnn_results = True
simulated_data_path = os.path.join(data_path, "output")

scenarios_list = None
if scenarios_list is None:
    scenarios_list = os.listdir(simulated_data_path)
total_scenarios = len(scenarios_list)

for i_sim in range(ini_simulations, end_simulations):
    msg = f"Working on simulation {i_sim + 1} out of {end_simulations}."
    logger.info(msg)
    for i_scenario, scenario_path in enumerate(scenarios_list):
        msg = f"\tWorking on scenario_id {scenario_path}. {i_scenario + 1} out of {total_scenarios}."
        logger.info(msg)
        predict_fn_list = []
        labels_fns_list = []
        full_path = os.path.join(simulated_data_path, scenario_path)
        all_files = os.listdir(full_path)
        # Read the data
        colnames, X_train, X_validation, X_test, target_train, target_validation, target_test = read_data(i_sim, full_path)
        abscissa_points = get_abscissa_points(colnames)
        X_full = np.row_stack((X_train, X_validation))
        target_full = np.row_stack((target_train, target_validation))
        # Transform the data
        X_train_grid = FDataGrid(
            data_matrix=X_train,
            grid_points=abscissa_points,
        )
        X_validation_grid = FDataGrid(
            data_matrix=X_validation,
            grid_points=abscissa_points,
        )
        X_test_grid = FDataGrid(
            data_matrix=X_test,
            grid_points=abscissa_points,
        )
        X_full_grid = FDataGrid(
            data_matrix=X_full,
            grid_points=abscissa_points,
        )
        ########## Linear model
        if get_lm_results:
            msg = f"\t\tFitting linear model"
            logger.info(msg)
            hyperopt_lm = HyperOptScikitFda(
                LinearRegression,
                abscissa_points=abscissa_points,
                domain_range=domain_range,
            )
            n_basis_list = list(range(4, 30, 1))
            hist_lm = hyperopt_lm.search(
                X_train=X_train,
                y_train=target_train[:, 0],
                X_val=X_validation,
                y_val=target_validation[:, 0],
                basis=BSplineBasis,
                n_basis_list=n_basis_list,
            )
            best_score_lm_list = [x.best_score_ for x in hist_lm]
            position_best_score_lm = np.argmax(best_score_lm_list)
            best_n_basis_lm = n_basis_list[position_best_score_lm]
            best_params_lm = hist_lm[position_best_score_lm].best_params_
            best_model_lm = hyperopt_lm.cls_estimator(**best_params_lm)
            best_basis_lm = BSplineBasis(
                n_basis=best_n_basis_lm,
                domain_range=domain_range,
            )
            X_full_bspline = X_full_grid.to_basis(best_basis_lm)
            X_test_bspline = X_test_grid.to_basis(best_basis_lm)
            _ = best_model_lm.fit(X_full_bspline, target_full[:, 0])
            # Transform predict function to use a numpy array as input
            predict_best_model_lm = predict_np_reshape(
                grid_points=abscissa_points,
                domain_range=domain_range,
                basis=X_full_bspline.basis,
                predict_fn=best_model_lm.predict,
            )
            predict_fn_list.append(predict_best_model_lm)
            labels_fns_list.append("lm")
            # Compute r^2
            predicted_test_lm = predict_best_model_lm(X_test)
            r2_test_lm = obtain_score(predicted_test_lm, target_test)
            r2_test_lm_name = f"r2_test_lm_{i_sim}.pkl"
            r2_test_lm_file = os.path.join(data_path, "output", scenario_path, r2_test_lm_name)
            with open(r2_test_lm_file, 'wb') as f_r2_lm:
                    pickle.dump(r2_test_lm, f_r2_lm)
        ########## KNN
        if get_knn_results:
            msg = "\t\tFitting knn model"
            logger.info(msg)
            hyperopt_knn = HyperOptScikitFda(
                KNeighborsRegressor,
                abscissa_points=abscissa_points,
                domain_range=domain_range,
            )
            hist_knn = hyperopt_knn.search(
                params={"n_neighbors": range(3, 30, 1)},
                X_train=X_train,
                y_train=target_train,
                X_val=X_validation,
                y_val=target_validation
            )
            best_score_knn_list = [x.best_score_ for x in hist_knn]
            position_best_score_knn = np.argmax(best_score_knn_list)
            best_params_knn = hist_knn[position_best_score_knn].best_params_
            best_model_knn = hyperopt_knn.cls_estimator(**best_params_knn)
            _ = best_model_knn.fit(X_full, target_full)
            predict_fn_list.append(best_model_knn.predict)
            labels_fns_list.append("knn")
            # Compute r^2
            predicted_test_knn = best_model_knn.predict(X_test)
            r2_test_knn = obtain_score(predicted_test_knn, target_test)
            r2_test_knn_name = f"r2_test_knn_{i_sim}.pkl"
            r2_test_knn_file = os.path.join(data_path, "output", scenario_path, r2_test_knn_name)
            with open(r2_test_knn_file, 'wb') as f_r2_knn:
                    pickle.dump(r2_test_knn, f_r2_knn)
        ########## FNN
        if get_fnn_results:
            msg = "\t\tFitting fnn model"
            logger.info(msg)
            hyperopt_fnn = HyperOptFnn(
                input_shape=(X_train.shape[1], 1),
                resolution=X_train.shape[1]
            )
            tuner_fnn = keras_tuner.RandomSearch(
                hyperopt_fnn,
                objective="val_loss",
                max_trials=max_trials_fnn,
                overwrite=True,
                directory=".",
                project_name="tune_hypermodel",
            )
            tuner_fnn.search(
                X_train,
                target_train,
                validation_data=(X_validation, target_validation),
                verbose=False,
            )
            best_params_fnn = tuner_fnn.get_best_hyperparameters()[0]
            best_epochs_fnn = best_params_fnn.get("epochs")
            best_batch_size_fnn = best_params_fnn.get("batch_size")
            hyperopt_best_fnn = HyperOptFnn(
                input_shape=(X_train.shape[1], 1),
                resolution=X_train.shape[1]
            )
            best_model_fnn = hyperopt_best_fnn.build(best_params_fnn)
            history_best_fnn = hyperopt_best_fnn.fit(
                hp=best_params_fnn,
                model=best_model_fnn,
                X=X_full,
                y=target_full,
                epochs=best_epochs_fnn,
                batch_size=best_batch_size_fnn,
                verbose=False
            )
            predict_best_model_fnn = predict_no_verbose(best_model_fnn.predict)
            predict_fn_list.append(predict_best_model_fnn)
            labels_fns_list.append("fnn")
            # Compute r^2
            predicted_test_fnn = best_model_fnn.predict(X_test, verbose=False)
            r2_test_fnn = obtain_score(predicted_test_fnn, target_test)
            r2_test_fnn_name = f"r2_test_fnn_{i_sim}.pkl"
            r2_test_fnn_file = os.path.join(data_path, "output", scenario_path, r2_test_fnn_name)
            with open(r2_test_fnn_file, 'wb') as f_r2_fnn:
                    pickle.dump(r2_test_fnn, f_r2_fnn)
        ########## Shapley value
        msg = "\t\tComputing Shapley value"
        logger.info(msg)
        shapley_fda_knn = ShapleyFda(
            X=X_test,
            abscissa_points=abscissa_points,
            target=target_test,
            domain_range=domain_range,
            verbose=False,
        )
        shapley_value = shapley_fda_knn.compute_shapley_value(
            num_permutations=num_permutations,
            predict_fns=predict_fn_list,
            labels_fns=labels_fns_list,
            num_intervals=num_intervals,
        )
        shapley_value_name = f"shapley_{i_sim}.pkl"
        values_shapley_file = os.path.join(data_path, "output", scenario_path, shapley_value_name)
        with open(values_shapley_file, 'wb') as f_val_shapley:
            pickle.dump(shapley_value, f_val_shapley)
