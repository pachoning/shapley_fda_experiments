from scipy.stats import beta
from skfda.misc import inner_product
from skfda.representation.basis import FourierBasis
from skfda.representation.grid import FDataGrid
from sklearn.utils import check_random_state
import numpy as np

class FdaSimulator:
    def __init__(
        self,
        ini=0,
        end=1,
        period=np.pi/2,
        step=0.01,
    ):
        self.ini = ini
        self.end = end
        self.domain_range = (self.ini, self.end)
        self.period = period
        self.step = step
        self.abscissa_points = np.arange(self.ini, self.end + self.step, self.step)
        self.total_abscissa_points = self.abscissa_points.shape[0]
        self.type_covariate = ["brownian_trend", "fourier_expansion", "symmetric_fourier_expanansion"]
        self.type_target = [
            "linear_unimodal",
            "linear_bimodal",
            "non_linear_unimodal",
            "non_linear_bimodal",
            "linear_discrete"
        ]

    def to_fdata_grid(self, X):
        X_fdata_grid = FDataGrid(
            data_matrix=X,
            grid_points=self.abscissa_points,
            domain_range=self.domain_range
        )
        return X_fdata_grid

    def get_brownian_motion_with_trend(
        self,
        sample_size,
        intercept,
        slope,
        random_state=None,
    ):
        random_state_gen = check_random_state(random_state)
        # Generate normal observations
        data = random_state_gen.normal(
            size=(sample_size, self.total_abscissa_points),
            loc=0,
            scale=1
        )
        data = np.divide(data, np.sqrt(self.total_abscissa_points))
        brownian = np.cumsum(data, axis=1)
        trend = np.reshape(
            np.add(intercept, np.multiply(slope, self.abscissa_points)),
            newshape=(1, -1)
        )
        col_vector_ones_sample_size = np.ones(shape=(sample_size, 1))
        trend_matrix = np.matmul(
            col_vector_ones_sample_size,
            trend
        )
        brownian_trend = np.add(brownian, trend_matrix)
        return brownian_trend

    def get_serie_representation(
        self,
        sample_size,
        wanted_symmetry,
        n_basis_simulated_data,
        sd_x,
        random_state=None,
    ):
        if n_basis_simulated_data % 2 == 0:
            n_basis_simulated_data = n_basis_simulated_data + 1
        basis_fourier = FourierBasis(
            n_basis=n_basis_simulated_data,
            period=self.period,
            domain_range=self.domain_range
        )
        row_vector_ones_total_abscissa_points = np.ones(shape=(1, self.total_abscissa_points))
        X_simulated = np.empty(shape=(sample_size, self.total_abscissa_points))
        basis_fourier_evaluated = np.squeeze(basis_fourier(self.abscissa_points))
        #lambda_coefficients = np.array([1/i for i in range(1, n_basis_simulated_data + 1)], ndmin=2)
        lambda_coefficients = np.array([1 for i in range(1, n_basis_simulated_data + 1)], ndmin=2)
        lambda_matrix = np.dot(lambda_coefficients.T, row_vector_ones_total_abscissa_points)
        for i in range(sample_size):
            if not random_state is None:
                random_state = random_state + i
            random_state_gen = check_random_state(random_state)
            normal_vector = random_state_gen.normal(scale=sd_x, size=(1, n_basis_simulated_data))
            normal_matrix = np.dot(normal_vector.T, row_vector_ones_total_abscissa_points)
            # Each basis is multiplied by the same coefficient. Therefore, given a basis (a row), we use
            # the same coefficient for all the columns (time)
            coefficients_basis_matrix = np.multiply(normal_matrix, lambda_matrix)
            basis_with_coefficients_matrix = np.multiply(basis_fourier_evaluated, coefficients_basis_matrix)
            sum_basis = np.sum(basis_with_coefficients_matrix, axis=0)
            X_simulated[i, :] = sum_basis
        if wanted_symmetry:
            X_simulated = X_simulated + np.flip(X_simulated, axis=1)
        return X_simulated

    def get_covariate(
        self,
        type_covariate,
        sample_size,
        **kwargs,
    ):
        data = None
        if not type_covariate in self.type_covariate:
            type_covariate_str = ', '.join(self.type_covariate)
            raise ValueError(f"type_covariate (for covariate) must be a value in {type_covariate_str}")
        random_state = kwargs["random_state"] if "random_state" in kwargs.keys() else None
        if type_covariate == "brownian_trend":
            intercept = kwargs["intercept_brownian"]
            slope = kwargs["slope_brownian"]
            data = self.get_brownian_motion_with_trend(
                sample_size=sample_size,
                intercept=intercept,
                slope=slope,
                random_state=random_state
            )
        elif type_covariate in ["fourier_expansion", "symmetric_fourier_expanansion"]:
            wanted_symmetry = type_covariate == "symmetric_fourier_expanansion"
            n_basis_simulated_data = kwargs["n_basis_simulated_data"]
            sd_x = kwargs["sd_x"]
            data = self.get_serie_representation(
                sample_size=sample_size,
                wanted_symmetry=wanted_symmetry,
                n_basis_simulated_data=n_basis_simulated_data,
                sd_x=sd_x,
                random_state=random_state,
            )
        return data

    def get_beta(
        self,
        alpha_param,
        beta_param,
        wanted_bimodal,
    ):
        # Build beta distribution data
        beta_distr = beta(alpha_param, beta_param)
        beta_pdf_abscissa = np.reshape(beta_distr.pdf(self.abscissa_points), newshape=(1, -1))
        if wanted_bimodal:
            beta_distr_other = beta(beta_param, alpha_param)
            beta_other_pdf_abscissa = np.reshape(beta_distr_other.pdf(self.abscissa_points), newshape=(1, -1))
            beta_pdf_abscissa = 0.5 * (beta_pdf_abscissa + beta_other_pdf_abscissa)
        return beta_pdf_abscissa

    def linear_discrete(
        self,
        data_input,
        positions,
    ):
        if len(positions) != 4:
            raise ValueError(
                "When type_transformation is linear_discrete, positions must be a list of four numbers"
            )
        total_columns = data_input.shape[1]
        col_indexes = [int(np.floor(x * total_columns)) for x in positions]
        data_input_filtered = data_input[:, col_indexes]
        x_0_2 = np.add(
            data_input_filtered[:, 0],
            np.abs(data_input_filtered[:, 2])
        )
        x_1_3 = np.multiply(
            np.power(data_input_filtered[:, 1], 2),
            data_input_filtered[:, 3]
        )
        data_output = np.add(x_0_2, x_1_3)
        data_output = np.reshape(
            data_output,
            newshape=(-1, 1)
        )
        return data_output, col_indexes

    def non_linerar_fn(
        self,
        data_input,
        alpha_param,
        beta_param,
        wanted_bimodal,
    ):
        beta_data = self.get_beta(
            alpha_param=alpha_param,
            beta_param=beta_param,
            wanted_bimodal=wanted_bimodal,
        )

        sample_size = data_input.shape[0]
        ones_vector_column = np.full(
            shape = (sample_size, 1),
            fill_value=1
        )
        beta_data_matrix = np.matmul(ones_vector_column, beta_data)
        data_input_flip = np.flip(
            data_input,
            axis=1,
        )
        matrix_product = np.multiply(data_input, beta_data_matrix)
        matrix_product_flip = np.multiply(data_input_flip, beta_data_matrix)
        matrix_stack = np.column_stack(
            (
                np.abs(matrix_product),
                np.abs(matrix_product_flip)
            )
        )
        data_output = np.max(matrix_stack, axis=1)
        data_output = np.reshape(
            data_output,
            newshape=(-1, 1)
        )
        return data_output, beta_data

    def linear_fn(
        self,
        data_input,
        alpha_param,
        beta_param,
        wanted_bimodal,
    ):
        data_input_grid = self.to_fdata_grid(X=data_input)
        beta_data = self.get_beta(
            alpha_param=alpha_param,
            beta_param=beta_param,
            wanted_bimodal=wanted_bimodal,
        )
        beta_data_grid = self.to_fdata_grid(X=beta_data)
        data_output = inner_product(data_input_grid, beta_data_grid)
        data_output = np.reshape(
            data_output,
            newshape=(-1, 1)
        )
        return data_output, beta_data

    def transform_data(
        self,
        type_transformation,
        data_input,
        **kwargs,
    ):
        data_output = None
        beta_data = None
        col_indexes_ld = None
        if not type_transformation in self.type_target:
            type_targete_str = ', '.join(self.type_target)
            raise ValueError(f"type_transformation (for target) must be a value in {type_targete_str}")

        if type_transformation in ["linear_unimodal", "linear_bimodal", "non_linear_unimodal", "non_linear_bimodal"]:
            alpha_param = kwargs["alpha_param"]
            beta_param = kwargs["beta_param"]
            wanted_bimodal = type_transformation in ["linear_bimodal", "non_linear_bimodal"]
            is_linear = type_transformation in ["linear_unimodal", "linear_bimodal"]
            fn_to_apply = self.linear_fn if is_linear else self.non_linerar_fn
            data_output, beta_data = fn_to_apply(
                data_input=data_input,
                alpha_param=alpha_param,
                beta_param=beta_param,
                wanted_bimodal=wanted_bimodal,
            )
        elif type_transformation == "linear_discrete":
            positions = kwargs["positions"]
            data_output, col_indexes_ld = self.linear_discrete(
                data_input=data_input,
                positions=positions,
            )
        return data_output, beta_data, col_indexes_ld

    def simulate(
        self,
        type_covariate,
        type_transformation,
        sample_size,
        eta,
        datasets_type=None,
        **kwargs,
    ):
        if eta <= 0 or eta >= 1:
            raise ValueError(f"eta must be inside the interval (0, 1)")
        covariate_list, phi_covariate_list, epsilon_list = [], [], []
        beta_data_list, col_indexes_ld_list, target_list = [], [], []

        if datasets_type is None:
            n_replicas = 1
        else:
            n_replicas = len(datasets_type)
        # Obtain the covariate
        for i_replica in range(n_replicas):
            covariate = self.get_covariate(
                type_covariate=type_covariate,
                sample_size=sample_size,
                **kwargs,
            )
            covariate_list.append(covariate)
            # Transform the covariate
            phi_covariate, beta_data, col_indexes_ld = self.transform_data(
                data_input=covariate,
                type_transformation=type_transformation,
                **kwargs,
            )
            phi_covariate_list.append(phi_covariate)
            var_phi_covariate = np.var(phi_covariate)
            var_epsilon = var_phi_covariate * (eta/(1 - eta))
            random_state = kwargs["random_state"] if "random_state" in kwargs.keys() else None
            if random_state:
                random_state = random_state + i_replica
            random_state_gen = check_random_state(random_state)
            epsilon = random_state_gen.normal(
                size=phi_covariate.shape,
                loc=0,
                scale=np.sqrt(var_epsilon),
            )
            epsilon_list.append(epsilon)
            target = np.add(phi_covariate, epsilon)
            beta_data_list.append(beta_data)
            col_indexes_ld_list.append(col_indexes_ld)
            target_list.append(target)
        return covariate_list, phi_covariate_list, epsilon_list, beta_data_list, col_indexes_ld_list, target_list
