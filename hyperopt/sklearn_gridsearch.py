from skfda.representation.basis import FDataBasis
from skfda.representation.grid import FDataGrid
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import numpy as np

class HyperOptScikitFda:
    def __init__(self, cls_estimator, abscissa_points, domain_range):
        self.cls_estimator = cls_estimator
        self.abscissa_points = abscissa_points
        self.domain_range = domain_range

    def search(self, params, X_train, y_train, X_val, y_val):
        model = self.cls_estimator()
        
        if isinstance(X_train, FDataBasis):
            X_train_np = X_train(self.abscissa_points)
            X_val_np = X_val(self.abscissa_points)
            X_np = np.concatenate((X_train_np, X_val_np))
            X_grid = FDataGrid(
                data_matrix=X_np,
                grid_points=self.abscissa_points,
                domain_range=self.domain_range
            )
            X = X_grid.to_basis(X_train.basis)
        else:
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
