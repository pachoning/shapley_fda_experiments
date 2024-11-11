from images import images_path
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.representation.grid import FDataGrid
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

def predict_from_np(grid_points, domain_range, basis, predict_fn):
    def inner_predict_from_np(X):
        X_to_grid = FDataGrid(data_matrix=X, grid_points=grid_points, domain_range=domain_range)
        X_to_basis = X_to_grid.to_basis(basis)
        prediction = predict_fn(X_to_basis)
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

def plot_shapley_function(
    obj,
    domain_range,
    translation_dict,
    plot_h_line=True,
    plot_v_line=True,
    display_legend=True,
):
    # Dividir por el ancho de cada intervalo de la particion
    # Equivalentea a multiplicar por el nmero de intervalos y dividir 
    # por el ancho del intervalo grande (domain_range)
    first_abscissa = domain_range[0]
    last_abscissa = domain_range[1]
    range_val = last_abscissa - first_abscissa
    x_min = [x[0] for x in obj["intervals"]]
    x_max = [x[1] for x in obj["intervals"]]
    factor = len(x_max)/range_val
    # First, we change the name of the keys
    # Round to 4 decimals the middle_points and rename the keys
    # To be deleted
    main_keys = ["intervals", "middle_points"]
    # Part of this, to be deleted
    new_obj = {}
    function_names = []
    for key in obj.keys():
        if key in main_keys:
            if key == "middle_points":
                new_obj[key] = [np.round(x, 4) for x in obj[key]]
            else:
                new_obj[key] = obj[key]
        else:
            new_key = translation_dict[key]
            new_obj[new_key] = obj[key]
    # Prepare the data to be plotted
    n_functions = len(new_obj.keys()) - len(main_keys)
    colors_code = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [0, 0, 0],
        [218, 112, 214],
        [107, 223, 205],
        [128, 128, 128]
    ])/255

    if n_functions <= colors_code.shape[0]:
        cmap = ListedColormap(colors_code)
    else:
        cmap = plt.cm.tab20
    colors = cmap.colors
    new_obj_keys = new_obj.keys()
    max_value = -np.inf
    i = 0
    for key in new_obj_keys:
        if not key in main_keys:
            current_data = np.array(new_obj[key])
            current_max = np.max(current_data)
            if current_max > max_value:
                max_value = current_max
            function_names.append(key)
            plt.hlines(
                factor * current_data,
                x_min,
                x_max,
                color=colors[i],
                linewidth=3
            )
            plt.step(
                x_min,
                factor * current_data,
                where="post",
                linestyle=(0, (1, 5)),
                color=colors[i],
            )
            i += 1
    if display_legend:
        plt.legend(
            handles=[Line2D([], [], color=col, lw=2.5) for col in colors],
            labels=function_names,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.05),
            fancybox=True,
            shadow=True,
            ncol=len(new_obj_keys),
        )
    if plot_h_line:
        plt.hlines(0, domain_range[0], domain_range[1], color='tab:grey')
    if plot_v_line:
        plt.vlines(domain_range[0], 0, factor * max_value, color='tab:grey')
    #plt.xlabel(x_lab)
    #plt.ylabel(y_lab)

def plot_shapley_value(
        obj,
        domain_range,
        translation_dict,
        plot_h_line=True,
        plot_v_line=True,
        display_legend=True,
        display_legend_below=True,
    ):
    # First, we change the name of the keys
    # Round to 4 decimals the middle_points and rename the keys
    # To be deleted
    lgd = None
    first_abscissa = domain_range[0]
    last_abscissa = domain_range[1]
    range_val = last_abscissa - first_abscissa
    main_keys = ["intervals", "middle_points"]
    x_max = [x[1] for x in obj["intervals"]]
    factor = len(x_max)/range_val
    # Part of this, to be deleted
    new_obj = {}
    function_names = []
    for key in obj.keys():
        if key in main_keys:
            if key == "middle_points":
                new_obj[key] = [np.round(x, 4) for x in obj[key]]
            else:
                new_obj[key] = obj[key]
        else:
            new_key = translation_dict[key]
            new_obj[new_key] = obj[key]

    # Prepare the data to be plotted
    n_functions = len(new_obj.keys()) - len(main_keys)
    colors_code = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [0, 0, 0],
        [218, 112, 214],
        [107, 223, 205],
        [128, 128, 128]
    ])/255

    if n_functions <= colors_code.shape[0]:
        cmap = ListedColormap(colors_code)
    else:
        cmap = plt.cm.tab20
    colors = cmap.colors

    new_obj_keys = new_obj.keys()
    max_value = -np.inf
    x_points = new_obj["middle_points"]
    total_points = len(x_points)
    x_points.insert(total_points, last_abscissa)
    x_points.insert(0, first_abscissa)

    i = 0
    for key in new_obj_keys:
        if not key in main_keys:
            function_names.append(key)
            current_data = np.array(new_obj[key])
            current_max = np.max(current_data)
            if current_max > max_value:
                max_value = current_max
            y_first_abscissa = current_data[0]
            y_last_abscissa = current_data[-1]
            current_data_mod = np.insert(current_data, total_points, y_last_abscissa)
            current_data_mod = np.insert(current_data_mod, 0, y_first_abscissa)
            plt.plot(
                x_points,
                factor * current_data_mod,
                color=colors[i],
                linewidth=3.0,
            )
            i += 1
    if display_legend:
        if display_legend_below:
            plt.legend(
                handles=[Line2D([], [], color=col, lw=2.5) for col in colors],
                labels=function_names,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.05),
                fancybox=True,
                shadow=True,
                ncol=len(new_obj_keys),
            )
        else:
            plt.legend(
                handles=[Line2D([], [], color=col, lw=2.5) for col in colors],
                labels=function_names,
                loc='upper left',
                bbox_to_anchor=(1.05, 1.0),
                fancybox=True,
                shadow=True,
            )
    if plot_h_line:
        plt.hlines(0, domain_range[0], domain_range[1], color='tab:grey')
    if plot_v_line:
        plt.vlines(domain_range[0], 0, factor * max_value, color='tab:grey')
    return lgd

def compute_mean_value(arr):
    mean_val = [np.mean(x) for x in arr]
    return mean_val

def obtain_values_from_object(shapley_object):
    return np.array([[x[1] for x in shapley_object]])
