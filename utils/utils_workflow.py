from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
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
