from data import data_path
from hyperopt.fnn import HyperOptFnn
from hyperopt.sklearn_gridsearch import HyperOptScikitFda
from shapley.shapley_fda import ShapleyFda
from skfda.ml.regression import KNeighborsRegressor, LinearRegression
from skfda.representation.basis import BSplineBasis
from skfda.representation.grid import FDataGrid
from utils.predict_np import predict_from_np
from utils.config import num_simulations
from utils.utils_workflow import get_abscissa_points, l2_reg, predict_no_verbose, read_data
import numpy as np
import os
import pickle


domain_range = (0, 1)
num_intervals = 20
num_permutations = 10
simulated_data_path = os.path.join(data_path, "output")
n_basis_representation = 10
basis_bsplines = BSplineBasis(
    n_basis=n_basis_representation,
    domain_range=(0 ,1)
)
num_intervals = 20
num_permutations = 10

reg_list = [l2_reg(x) for x in np.arange(0.2, 1.5, 0.2)]
reg_list.append(None)

scenarios_list = os.listdir(simulated_data_path)
total_scenarios = len(scenarios_list)

for i_sim in range(num_simulations):
    print(f"Working on simulation {i_sim + 1} out of {num_simulations}")
    for i_scenario, scenario_path in enumerate(scenarios_list):
        print(f"\tWorking on scenario_id {scenario_path}. {i_scenario + 1} out of {total_scenarios}")
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
        X_train_bspline = X_train_grid.to_basis(basis_bsplines)
        X_validation_bspline = X_validation_grid.to_basis(basis_bsplines)
        X_test_bspline = X_test_grid.to_basis(basis_bsplines)
        X_full_bspline = X_full_grid.to_basis(basis_bsplines)
        ########## Linear model
        print("\t\tFitting linear model")
        hyperopt_lm = HyperOptScikitFda(
            LinearRegression,
            abscissa_points=abscissa_points,
            domain_range=domain_range,
        )
        params_lm = {
            "regularization": reg_list
        }
        hist_lm = hyperopt_lm.search(
            params=params_lm,
            X_train=X_train_bspline,
            y_train=target_train[:, 0],
            X_val=X_validation_bspline,
            y_val=target_validation[:, 0]
        )
        best_params_lm = hist_lm.best_params_
        best_model_lm = hyperopt_lm.cls_estimator(**best_params_lm)
        _ = best_model_lm.fit(X_full_bspline, target_full[:, 0])
        # Transform predict function to use a numpy array as input
        pred_lm = predict_from_np(
            grid_points=abscissa_points,
            domain_range=domain_range,
            basis=X_full_bspline.basis,
            predict_fn=best_model_lm.predict
        )
        # Shapley for the Linear Model
        print("\t\tComputing Shapley for linear model")
        shapley_fda_lm = ShapleyFda(
            predict_fn=pred_lm,
            X=X_test,
            abscissa_points=abscissa_points,
            target=target_test[:, 0],
            domain_range=domain_range,
            verbose=False,
        )
        values_shapley_lm = shapley_fda_lm.compute_shapley_value(
            num_intervals=num_intervals,
            num_permutations=num_permutations,
        )
        values_shapley_lm_name = f"shapley_lm_{i_sim}.pkl"
        values_shapley_lm_file = os.path.join(data_path, "output", scenario_path, values_shapley_lm_name)
        with open(values_shapley_lm_file, 'wb') as f_lm:
                pickle.dump(values_shapley_lm, f_lm)
        ########## KNN
        print("\t\tFitting knn model")
        hyperopt_knn = HyperOptScikitFda(
             KNeighborsRegressor,
             abscissa_points=abscissa_points,
             domain_range=domain_range,
        )
        hist_knn = hyperopt_knn.search(
             params={"n_neighbors": range(3, 12, 2)},
             X_train=X_train,
             y_train=target_train,
             X_val=X_validation,
             y_val=target_validation
        )
        best_params_knn = hist_knn.best_params_
        best_model_knn = hyperopt_knn.cls_estimator(**best_params_knn)
        _ = best_model_knn.fit(X_full, target_full)
        # Shapley for the KNN
        print("\t\tComputing Shapley for knn model")
        shapley_fda_knn = ShapleyFda(
             predict_fn=best_model_knn.predict,
             X=X_test,
             abscissa_points=abscissa_points,
             target=target_test,
             domain_range=domain_range,
             verbose=False,
        )
        values_shapley_knn = shapley_fda_knn.compute_shapley_value(
             num_intervals=num_intervals,
             num_permutations=num_permutations,
        )
        values_shapley_knn_name = f"shapley_knn_{i_sim}.pkl"
        values_shapley_knn_file = os.path.join(data_path, "output", scenario_path, values_shapley_knn_name)
        with open(values_shapley_knn_file, 'wb') as f_knn:
            pickle.dump(values_shapley_knn, f_knn)
        
        ########## FNN
        print("\t\tFitting fnn model")
        hyperopt_fnn = HyperOptFnn(
            input_shape=(X_train.shape[1], 1),
            resolution=X_train.shape[1]
        )

        tuner_fnn = hyperopt_fnn.build_tuner(
            objective="val_loss",
            max_trials=6,
            overwrite=True,
            directory=".",
            project_name="tune_hypermodel",
        )

        tuner_fnn.search(
            X_train,
            target_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_validation, target_validation),
            verbose=False,
        )

        best_model_fnn = HyperOptFnn(
            input_shape=(X_train.shape[1], 1),
            resolution=X_train.shape[1]
        )

        best_params_fnn = tuner_fnn.get_best_hyperparameters(5)[0]
        model_fnn = best_model_fnn.build(best_params_fnn)
        history = best_model_fnn.fit(
            best_params_fnn,
            model_fnn,
            X_full,
            target_full,
            batch_size=32,
            epochs=1,
            verbose=False,
        )
        # Shapley for FNN
        print("\t\tComputing Shapley for fnn model")
        shapley_fda_fnn = ShapleyFda(
            predict_fn=predict_no_verbose(model_fnn.predict),
            X=X_test,
            abscissa_points=abscissa_points,
            target=target_test,
            domain_range=domain_range,
            verbose=False,
        )

        values_shapley_fnn = shapley_fda_fnn.compute_shapley_value(
            num_intervals=num_intervals,
            num_permutations=num_permutations,
        )
        values_shapley_fnn_name = f"shapley_fnn_{i_sim}.pkl"
        values_shapley_fnn_file = os.path.join(data_path, "output", scenario_path, values_shapley_fnn_name)
        with open(values_shapley_fnn_file, 'wb') as f_knn:
            pickle.dump(values_shapley_fnn, f_knn)
        