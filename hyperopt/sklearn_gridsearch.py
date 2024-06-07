import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit

class HyperOptScikitFda:
    def __init__(self, cls_estimator):
        self.cls_estimator = cls_estimator

    def search(self, params, X_train, y_train, X_val, y_val):
        model = self.cls_estimator()
        X = np.concatenate((X_train, X_val))
        y = np.concatenate((y_train, y_val))
        zeros_vector = np.zeros(X_train.shape[0])
        ones_vector = np.ones(X_val.shape[0])
        all_vector = np.concatenate((zeros_vector, ones_vector))
        ps = PredefinedSplit(all_vector)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=params,
            cv=ps,
            scoring="neg_mean_squared_error",
            return_train_score=True,
        )
        return grid_search.fit(X, y)

    def build_estimator(self, params):
        return self.cls_estimator(**params)
