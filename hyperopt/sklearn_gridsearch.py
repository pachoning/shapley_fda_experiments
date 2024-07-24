from skfda.representation.basis import FDataBasis
from skfda.representation.grid import FDataGrid
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import numpy as np

class HyperOptScikitFda:
    def __init__(self, cls_estimator, abscissa_points, domain_range):
        self.cls_estimator = cls_estimator
        self.abscissa_points = abscissa_points
        self.domain_range = domain_range

    def search(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        params=None,
        basis=None,
        n_basis_list=[],
    ):
        results = []
        model = self.cls_estimator()
        X_np = np.concatenate((X_train, X_val))
        y = np.concatenate((y_train, y_val))
        train_vector = np.full(shape=X_train.shape[0], fill_value=-1, dtype=np.int32)
        validation_vector = np.full(shape=X_val.shape[0], fill_value=0, dtype=np.int32)
        all_vector = np.concatenate((train_vector, validation_vector))
        ps = PredefinedSplit(all_vector)
        if basis is None:
            total_basis = 1
        else:
            total_basis = len(n_basis_list)
        i_basis = 0
        while i_basis < total_basis:
            if basis:
                n_basis = n_basis_list[i_basis]
                basis_representation = basis(
                    n_basis=n_basis,
                    domain_range=self.domain_range,
                )
                X_grid = FDataGrid(
                    data_matrix=X_np,
                    grid_points=self.abscissa_points,
                    domain_range=self.domain_range
                )
                X = X_grid.to_basis(basis_representation)
            else:
                X = X_np.copy()
            if params is None:
                params = {}
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=ps,
                scoring="neg_mean_squared_error",
                return_train_score=True,
                refit=False,
            )
            gs_results = grid_search.fit(X, y)
            results.append(gs_results)
            i_basis += 1
        return results
