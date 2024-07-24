from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.representation.grid import FDataGrid
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def read_data(i_sim, path):
    type_data = ["train", "validation", "test"]
    X_str = "X_sim_{}_{}.csv"
    target_str = "target_sim_{}_{}.csv"
    X = [
            pd.read_csv(os.path.join(path, X_str.format(x, i_sim))) for x in type_data
        ]
    target = [
            pd.read_csv(os.path.join(path, target_str.format(x, i_sim))) for x in type_data
        ]
    
    colnames = X[0].columns.values
    all_data = [*X, *target]
    all_data_numpy = [x.to_numpy() for x in all_data]
    return [colnames, *all_data_numpy]

def get_abscissa_points(names):
    points = [float(c.split("_")[1]) for c in names]
    return np.array(points)

def predict_no_verbose(predict_fn):
    def inner(*args, **kwargs):
        return predict_fn(*args, verbose=False, **kwargs)
    return inner

def predict_no_verbose_squeeze(predict_fn):
    def inner(*args, **kwargs):
        predicted_value = predict_fn(*args, verbose=False, **kwargs)
        return predicted_value[:, 0]
    return inner

def predict_np_reshape(grid_points, domain_range, basis, predict_fn):
    def inner_predict_from_np(X):
        X_to_grid = FDataGrid(data_matrix=X, grid_points=grid_points, domain_range=domain_range)
        X_to_basis = X_to_grid.to_basis(basis)
        prediction = predict_fn(X_to_basis)
        if len(prediction.shape) == 1:
            prediction = np.reshape(prediction, newshape=(-1, 1))
        return prediction
    return inner_predict_from_np

def l2_reg(lambda_value):
    operator = L2Regularization(
        linear_operator=LinearDifferentialOperator(2),
        regularization_parameter=lambda_value
    )
    return operator

def obtain_score(prediction, target):
    diff_target_pred = np.subtract(target, prediction)
    diff_target_pred_sq = np.power(diff_target_pred, 2)
    rss = np.sum(diff_target_pred_sq)
    target_mean = np.mean(target)
    diff_target_target_mean = np.subtract(target, target_mean)
    diff_target_target_mean_sq = np.power(diff_target_target_mean, 2)
    tss = np.sum(diff_target_target_mean_sq)
    r2 = 1 - rss/tss
    return r2

def plot_step_fn(
    x_min,
    x_max,
    values,
    colors,
    models,
    domain_range,
    x_lab,
    y_lab,
    plot_continuous_vline=False,
    plt_h_line=True,
    plot_v_line=True,
):
    # Dividir por el ancho de cada intervalo de la particion
    # Equivalentea a multiplicar por el nmero de intervalos y dividir por el ancho del intervalo grande (domain_range)
    range_val = domain_range[1] - domain_range[0]
    factor = len(x_max)/range_val
    print("Num intervals:", len(x_max))
    print("Range value:", range_val)
    print("Factor:", factor)
    #factor = 1
    n_models = len(models)
    if plot_continuous_vline:
        plt.step(
            x_min,
            factor * values.T,
            where="post",
            label=models
        )
    else:
        plt.step(
        x_min,
        factor * values.T,
        where="post",
        linestyle=(0, (1, 5)),
        label=models
    )
    plt.legend(handles=[mpatches.Patch(color=col, label=mod) for col, mod in zip(colors, models)])
    for i in range(n_models):
        data_i = factor * values[i, :]
        plt.hlines(
            data_i,
            x_min,
            x_max,
            color=colors[i],
            linewidths=3
        )
    if plt_h_line:
        plt.hlines(0, domain_range[0], domain_range[1], color='tab:grey')
    if plot_v_line:
        plt.vlines(0, 0, factor * np.max(values), color='tab:grey')
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)

def compute_mean_value(arr):
    mean_val = [np.mean(x) for x in arr]
    return mean_val

def obtain_values_from_object(shapley_object):
    return np.array([[x[1] for x in shapley_object]])
