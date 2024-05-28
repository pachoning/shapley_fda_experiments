from skfda.representation.grid import FDataGrid

def predict_from_np(grid_points, domain_range, basis, predict_fn):
    def inner_predict_from_np(X):
        X_to_grid = FDataGrid(data_matrix=X, grid_points=grid_points, domain_range=domain_range)
        X_to_basis = X_to_grid.to_basis(basis)
        prediction = predict_fn(X_to_basis)
        return prediction
    return inner_predict_from_np
