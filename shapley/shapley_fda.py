from math import factorial
import numpy as np
import matplotlib.pyplot as plt


class ShapleyFda:
    def __init__(
        self,
        X,
        abscissa_points,
        target,
        domain_range,
        verbose,
    ):
        self.X = X
        self.abscissa_points = abscissa_points
        self.target = target
        self.domain_range = domain_range
        self.verbose = verbose
        self.shapley_values = None

    def validations(self, num_intervals, set_intervals):
        pass

    def print(
        self,
        *args,
    ):
        if self.verbose:
            str_print = ""
            for arg in args:
                str_print = str_print + " " + str(arg)
            print(str_print)

    def to_numpy(self, obj):
        obj_np = obj
        if isinstance(obj, int) or isinstance(obj, float):
            obj_np = np.array([obj])
        return obj_np

    def compute_function_from_matrix(self, set_abscissa_points, matrix):
        set_abscissa_points = self.to_numpy(set_abscissa_points)
        i_point = 0
        f_points = np.empty(shape=(matrix.shape[0], set_abscissa_points.shape[0]))
        for point in set_abscissa_points:
            if (point < self.domain_range[0] or point > self.domain_range[1]):
                raise ValueError("points contains a point outside the domain range (domain_range)")
            min_position = np.max(np.argwhere(point >= self.abscissa_points))
            num_min = self.abscissa_points[min_position]
            if (np.abs(num_min - point) < 1e-7):
                f_points[:, i_point] = matrix[:, min_position]
            else:
                max_position = np.min(np.argwhere(point < self.abscissa_points))
                num_max = self.abscissa_points[max_position]
                w_min = 1 - (point - num_min)/(num_max - num_min)
                w_max = 1 - (num_max - point)/(num_max - num_min)
                f_min = matrix[:, min_position]
                f_max = matrix[:, max_position]
                f_eval_point = np.add(np.multiply(w_min, f_min), np.multiply(w_max, f_max))
                f_points[:, i_point] = f_eval_point
            i_point += 1
        return f_points

    def compute_f(self, set_abscissa_points):
        return self.compute_function_from_matrix(set_abscissa_points, self.X)

    def create_set_intervals(self, num_intervals, intervals):
        if num_intervals is None and intervals is None:
            raise ValueError("Either num_intervals or intervals must not be None")
        elif num_intervals:
            ini_domain_range = self.domain_range[0]
            end_domain_range = self.domain_range[1]
            long_domain_range = end_domain_range - ini_domain_range
            intervals_lower_bound = np.array(
                [ini_domain_range + i * long_domain_range/num_intervals for i in range(num_intervals)]
            )
            intervals_upper_bound = np.array(
                [ini_domain_range + (i + 1) * long_domain_range/num_intervals for i in range(num_intervals)]
            )
            intervals = np.stack((intervals_lower_bound, intervals_upper_bound), axis=1)
        elif intervals:
            # TODO: if the user provides intervals, standardise it so that
            # it has the same shape as if it were created by the previous statement
            pass
        return intervals

    def create_permutations(self, num_intervals, num_permutations, seed):
        if num_permutations % 2 != 0:
            num_permutations = num_permutations + 1
        set_permutations = set()
        total_set_permutations = len(set_permutations)
        # Error when impossible number of permutations is desired
        if num_permutations > factorial(num_intervals):
            raise ValueError("num_permutations can no be greater than the factorial of number of intervals")
        # Iterate to get half of the permutations
        i_seed = 10
        while total_set_permutations < num_permutations//2:
            if seed:
                np.random.seed(seed + i_seed)
            permutation = np.random.choice(a=num_intervals, size=num_intervals, replace=False)
            permutation_sym = np.subtract(num_intervals - 1, permutation)
            permutation_tuple = tuple(permutation)
            permutation_sym_tuple = tuple(permutation_sym)
            if not (permutation_tuple in set_permutations or permutation_sym_tuple in set_permutations):
                set_permutations.add(permutation_tuple)
                set_permutations.add(permutation_sym_tuple)
            total_set_permutations = len(set_permutations)
            i_seed += 1
        # Complete with symmetric permutations
        return set_permutations

    def break_permutation(self, permutation, global_interval_position, use_interval):
        permutation_array = np.array(permutation)
        interval_position_inside_permutation = np.argwhere(global_interval_position == permutation_array).squeeze()
        # Given the permutation, some we will have to interpolate the information for some of the intervals.
        # Depending if the current interval is used or not.
        if use_interval:
            available_intervals = permutation_array[:(interval_position_inside_permutation + 1)]
            non_available_intervals = permutation_array[(interval_position_inside_permutation + 1):]
        else:
            available_intervals = permutation_array[:interval_position_inside_permutation]
            non_available_intervals = permutation_array[interval_position_inside_permutation:]
        return available_intervals, non_available_intervals

    def map_abscissa_interval(self, set_intervals):
        set_intervals_shape = set_intervals.shape
        map_object = np.full(shape=self.abscissa_points.shape, fill_value=1, dtype=int)
        num_intervals = set_intervals_shape[0]
        last_end_interval = set_intervals[num_intervals-1, 1]
        i_abscissa = 0
        for abscissa in self.abscissa_points:
            if(np.abs(abscissa - last_end_interval) < 1e-7):
                interval_position = num_intervals - 1
            else:
                interval_position = np.ravel(np.argwhere((abscissa >= set_intervals[:, 0]) & (abscissa < set_intervals[:, 1])))
                interval_position = interval_position[0]
            map_object[i_abscissa] = interval_position
            i_abscissa += 1
        return map_object

    def get_abscissa_from_intervals(self, intervals, mapping_abscissa_interval):
        set_abscissa = []
        if isinstance(intervals, int):
            intervals = [intervals]
        for interval in intervals:
            abscissa_interval = np.ravel(np.argwhere(interval == mapping_abscissa_interval))
            set_abscissa.extend(abscissa_interval.tolist())
        return np.array(set_abscissa, dtype=np.int64)

    def conditional_expectation(
            self,
            mean_1,
            mean_2,
            matrix_mult
        ):
        def inner_conditional_expectation(x_2):
            diff = np.subtract(x_2, mean_2)
            vector_mult = np.dot(matrix_mult, diff)
            result = np.add(mean_1, vector_mult)
            return result
        return inner_conditional_expectation

    def recompute_covariate(
            self,
            available_intervals,
            non_available_intervals,
            mapping_abscissa_interval,
            mean_f,
            covariance_f
        ):
        recomputed_covariate = np.empty(shape=self.X.shape)
        # For available_intervals, use real values
        position_available_abscissa = self.get_abscissa_from_intervals(available_intervals, mapping_abscissa_interval)
        available_abscissa = self.abscissa_points[position_available_abscissa]
        f_available_abscissa = self.compute_f(available_abscissa)
        recomputed_covariate[:, position_available_abscissa] = f_available_abscissa
        # For non_available_intervals, use the conditional expectation
        position_non_available_abscissa = self.get_abscissa_from_intervals(
            non_available_intervals,
            mapping_abscissa_interval
        )
        non_available_abscissa = self.abscissa_points[position_non_available_abscissa]
        self.print(
            "\t\tavailable_abscissa:",
            available_abscissa.tolist(),
            "non_available_abscissa",
            non_available_abscissa.tolist()
        )
        # Get main statistics to compute conditional expetation
        mean_available_abscissa = mean_f[position_available_abscissa]
        mean_non_available_abscissa = mean_f[position_non_available_abscissa]
        covariance_mix = covariance_f[position_non_available_abscissa, :][:, position_available_abscissa]
        covariance_available_abscissa = covariance_f[position_available_abscissa, :][:, position_available_abscissa]
        det = np.linalg.det(covariance_available_abscissa)

        max_eigenvalue = None
        min_eigenvalue = None
        eigen_ration = det
        if covariance_available_abscissa.shape[1] > 0:
            eigenvalues, eigenvectors = np.linalg.eig(covariance_available_abscissa)
            max_eigenvalue = np.max(np.abs(eigenvalues))
            min_eigenvalue = np.min(np.abs(eigenvalues))
            eigen_ration = min_eigenvalue/max_eigenvalue
        invertibility = min(det, eigen_ration)
        #self.print("\t\tdet_matrix", det_matrix)
        self.print("\t\tmax min", max_eigenvalue, min_eigenvalue)
        if invertibility < 1e-100:
            self.print("\t\tpseudo inversa!!!!")
            inv_covariance_available_abscissa = np.linalg.pinv(covariance_available_abscissa)
        else:    
            self.print("\t\tinversa")
            inv_covariance_available_abscissa = np.linalg.inv(covariance_available_abscissa)
        matrix_mult = np.matmul(covariance_mix, inv_covariance_available_abscissa)
        conditional_expectation_fn = self.conditional_expectation(
            mean_1=mean_non_available_abscissa,
            mean_2=mean_available_abscissa,
            matrix_mult=matrix_mult
        )
        total_individuals = self.X.shape[0]
        # TODO: review this part to speed it up. Mw may use the method compute_f
        for i in range(total_individuals):
            X_i_available_abscissa = np.reshape(
                self.X[i][position_available_abscissa],
                newshape=(-1, 1)
            )
            conditional_expectation_i = np.ravel(conditional_expectation_fn(X_i_available_abscissa))
            recomputed_covariate[i][position_non_available_abscissa] = conditional_expectation_i
        return recomputed_covariate

    def obtain_score(self, covariate, target, predict_fn_list, labels_predict_fns_list):
        score_dict = {}
        for  predict_fn, label in zip(predict_fn_list, labels_predict_fns_list):
            prediction = predict_fn(covariate)
            diff_target_pred = np.subtract(target, prediction)
            diff_target_pred_sq = np.power(diff_target_pred, 2)
            rss = np.sum(diff_target_pred_sq)
            target_mean = np.mean(target)
            diff_target_target_mean = np.subtract(target, target_mean)
            diff_target_target_mean_sq = np.power(diff_target_target_mean, 2)
            tss = np.sum(diff_target_target_mean_sq)
            r2 = 1 - rss/tss
            r2_c = r2
            if r2_c < 0:
                r2_c = 0
            score_dict[label] = r2_c
        return score_dict

    def hash_array(self, array):
        sorted_array = np.sort(array)
        str_hash = ""
        for position, x in enumerate(sorted_array):
            if position == 0:
                str_hash = str(x)
            else:
                str_hash = str_hash + "_" + str(x)
        return str_hash

    def compute_model_based(
        self,
        hashed_available_intervals,
        available_intervals,
        non_available_intervals,
        mapping_abscissa_interval,
        available_intervals_computed,
        shapley_scores_computed,
        predict_fn_list,
        labels_predict_fns_list,
        mean_f,
        covariance_f,
        covariates_computed,
    ):
        # If the score is available for the set of available intervals, use it
        shapley_score = dict()
        total_predict_fn = len(predict_fn_list)
        if total_predict_fn > 0:
            if hashed_available_intervals in available_intervals_computed:
                for label in labels_predict_fns_list:
                    shapley_score[label] = shapley_scores_computed[label][hashed_available_intervals]
            else:
                # Recreate the set of functions without considering the interval
                covariate_recreated = self.recompute_covariate(
                    available_intervals,
                    non_available_intervals,
                    mapping_abscissa_interval,
                    mean_f,
                    covariance_f
                )
                # Compute the score
                shapley_score = self.obtain_score(
                    covariate_recreated,
                    self.target,
                    predict_fn_list,
                    labels_predict_fns_list
                )
                # Store info in cache
                covariates_computed[hashed_available_intervals] = covariate_recreated
                for label in labels_predict_fns_list:
                    shapley_scores_computed[label][hashed_available_intervals] = shapley_score[label]
        return shapley_score

    def  compute_mrmr_r2_based(
            self,
            hashed_available_intervals,
            available_intervals,
            mapping_abscissa_interval,
            available_intervals_computed,
            shapley_scores_computed,
        ):
        # In the information is already computed, use it
        if hashed_available_intervals in available_intervals_computed:
            score = shapley_scores_computed["mRMR_r2"][hashed_available_intervals]
        else:
            # Get the submatrix
            position_available_abscissa = self.get_abscissa_from_intervals(
                available_intervals,
                mapping_abscissa_interval
            )
            available_abscissa = self.abscissa_points[position_available_abscissa]
            total_available_abscissa = available_abscissa.shape[0]
            f_available_abscissa = self.compute_f(available_abscissa)
            redundancy = 0
            relevance = 0
            # Compute the metrics
            if total_available_abscissa == 0:
                relevance = 0
                redundancy = 1
            else:
                redundancy = np.mean(np.abs(np.corrcoef(f_available_abscissa, rowvar=False)))
                relevance_matrix = np.corrcoef(f_available_abscissa, self.target, rowvar=False)
                relevance = np.mean(np.abs(relevance_matrix[:, -1]))
            score = relevance/redundancy
            # Store the info in memory
            shapley_scores_computed["mRMR_r2"][hashed_available_intervals] = score
        return score

    def compute_mrmr_dist_corr_based(
            self,
            hashed_available_intervals,
            available_intervals,
            available_intervals_computed,
            shapley_scores_computed,
            H_matrix,
            dict_distance_diff_covariable,
            distance_target_centered,
            dist_covariance_yy,
        ):
        # If the information is in cache, use it
        if hashed_available_intervals in available_intervals_computed:
            score = shapley_scores_computed["mRMR_distance_correlation"][hashed_available_intervals]
        # If the information is not in cache, compute it
        else:
            score = 0
            total_available_intervals = available_intervals.shape[0]
            # If we are not considering the empty set, compute the score
            if total_available_intervals > 0:
                total_individuals = self.X.shape[0]
                sum_distance_diff_covariable = np.full(
                    shape=(total_individuals, total_individuals),
                    fill_value=0.0,
                    dtype="float32",
                )
                for current_interval in available_intervals:
                    distance_diff_current_interval = dict_distance_diff_covariable[current_interval]
                    sum_distance_diff_covariable = np.add(sum_distance_diff_covariable, distance_diff_current_interval)
                distance_covariate_centered = self.double_center_matix(
                    H=H_matrix,
                    D=sum_distance_diff_covariable,
                )
                dist_covariance_xy = self.compute_sq_root_mean_of_products(
                    X1=distance_covariate_centered,
                    X2=distance_target_centered,
                )
                dist_covariance_xx = self.compute_sq_root_mean_of_products(
                    X1=distance_covariate_centered,
                    X2=distance_covariate_centered,
                )
                denominator = np.sqrt(dist_covariance_xx * dist_covariance_yy)
                score = dist_covariance_xy/denominator
            shapley_scores_computed["mRMR_distance_correlation"][hashed_available_intervals] = score
        return score

    def compute_sq_root_mean_of_products(
        self,
        X1,
        X2,
    ):
        product = np.multiply(
            X1,
            X2,
        )
        result = np.sqrt(np.mean(product))
        return result

    def compute_l2_norm(
        self,
        abscissa_points,
        X,
    ):
        if len(X.shape) < 2:
            X = np.reshape(X, newshape=(1, -1))
        # Compute the square of the function
        X_sq = np.power(X, 2)
        # compute the lag of X
        X_sq_lag = np.roll(X_sq, shift=-1, axis=1)
        # [f(x1) + f(x2), ...., f(x_m) + f(x_m-1)]. We need to disregar the last position becuase
        # np.roll provides a matrix of the same size as f_available_abscissa.
        X_sq_sum = np.add(X_sq, X_sq_lag)
        X_sq_sum_filter = X_sq_sum[:, :-1]
        trapezoid_height = np.multiply(0.5, X_sq_sum_filter)
        # Now, we need to compute deltax = xk -x x_k-1
        abscissa_points_lag = np.roll(abscissa_points, shift=-1)
        diff_available_abscissa = np.subtract(abscissa_points_lag, abscissa_points)
        diff_available_filter = diff_available_abscissa[:-1]
        trapezoid_base = np.reshape(diff_available_filter, newshape=(-1, 1))
        # Area of the trapezoid
        area = np.matmul(trapezoid_height, trapezoid_base)
        # The squared norm is the integral. The integral is approximated by the area of the trapeziod
        l2_norm = np.sqrt(area)
        return l2_norm

    def double_center_matix(self, H, D):
        result = np.matmul(H, np.matmul(D, H))
        return result

    def compute_distance_difference(
        self,
        intervals,
        mapping_abscissa_interval,
    ):
        total_individuals = self.X.shape[0]
        total_intervals = intervals.shape[0]
        dict_distance_diff_cov = dict()
        distance_diff_target = np.full(
            shape=(total_individuals, total_individuals),
            fill_value=0.0,
            dtype="float32",
        )
        target = self.target
        if len(target.shape) < 2:
            target = np.reshape(target, newshape=(-1, 1))
        for i_interval in range(total_intervals):
            distance_matrix_covariate = np.full(
                shape=(total_individuals, total_individuals),
                fill_value=0.0,
                dtype="float32",
            )
            position_current_abscissa = self.get_abscissa_from_intervals(
                i_interval,
                mapping_abscissa_interval,
            )
            current_abcissa = self.abscissa_points[position_current_abscissa]
            f_current_abscissa = self.compute_f(current_abcissa)
            i_ind = 0
            # Compute distances matrices (covariate and target)
            while i_ind < total_individuals:
                f_i = f_current_abscissa[i_ind, :]
                target_i = np.squeeze(target[i_ind, :])
                j_ind = 0
                while j_ind < i_ind:
                    f_j = f_current_abscissa[j_ind, :]
                    target_j = np.squeeze(target[j_ind, :])
                    diff_covariate_ij = np.subtract(f_i, f_j)
                    distance_covariate_ij = self.compute_l2_norm(
                        abscissa_points=current_abcissa,
                        X=diff_covariate_ij,
                    )
                    distance_matrix_covariate[i_ind, j_ind] = distance_covariate_ij
                    distance_matrix_covariate[j_ind, i_ind] = distance_covariate_ij
                    # Fill in the target only once
                    if i_interval == 0:
                        distance_target_ij = np.abs(np.subtract(target_i, target_j))
                        distance_diff_target[i_ind, j_ind] = distance_target_ij
                        distance_diff_target[j_ind, i_ind] = distance_target_ij
                    j_ind += 1
                i_ind += 1
            dict_distance_diff_cov[i_interval] = distance_matrix_covariate
        return dict_distance_diff_cov, distance_diff_target

    def compute_interval_relevance(
        self,
        set_permutations,
        mapping_abscissa_interval,
        mean_f,
        covariance_f,
        interval_position,
        covariates_computed,
        available_intervals_computed,
        shapley_scores_computed,
        compute_mrmr_r2,
        compute_mrmr_distance_correlation,
        predict_fn_list,
        labels_predict_fns_list,
        dict_distance_diff_covariable,
        distance_target_centered,
        dist_covariance_yy,
        H_matrix,
    ):
        set_differences = dict()
        mean_value = dict()
        empty_dict = dict()
        #total_predict_fn = len(predict_fn_list)

        for label in labels_predict_fns_list:
            set_differences[label] = []
            mean_value[label] = np.full(shape=(), fill_value=np.nan)
            empty_dict[label] = dict()

        if compute_mrmr_r2:
            set_differences["mRMR_r2"] = []
            mean_value["mRMR_r2"] = np.full(shape=(), fill_value=np.nan)
            empty_dict["mRMR_r2"] = dict()

        if compute_mrmr_distance_correlation:
            set_differences["mRMR_distance_correlation"] = []
            mean_value["mRMR_distance_correlation"] = np.full(shape=(), fill_value=np.nan)
            empty_dict["mRMR_distance_correlation"] = dict()

        # For each permutation
        for i_permutation in set_permutations:
            self.print("\tPermutation:", i_permutation)
            shapley_score_permutation = empty_dict.copy()

            for use_interval in (False, True):
                # Break the permutation into two parts:
                    # first part is the one we are allowed to use
                    # second part is the one to be interpolated, i.e, non-available information
                available_intervals, non_available_intervals = self.break_permutation(
                    i_permutation,
                    interval_position,
                    use_interval
                )
                # Available intervals are hashed to store them in the cache
                hashed_available_intervals = self.hash_array(available_intervals)
                self.print(
                    "\t\tuse_interval:", use_interval,
                    "available_intervals:", available_intervals, 
                    "non_available_intervals:", non_available_intervals,
                    "hashed_available_intervals", hashed_available_intervals
                )
                # Compute Shapley value recreating the covariable
                model_based_shapley_score = self.compute_model_based(
                        hashed_available_intervals=hashed_available_intervals,
                        available_intervals=available_intervals,
                        non_available_intervals=non_available_intervals,
                        mapping_abscissa_interval=mapping_abscissa_interval,
                        available_intervals_computed=available_intervals_computed,
                        shapley_scores_computed=shapley_scores_computed,
                        predict_fn_list=predict_fn_list,
                        labels_predict_fns_list=labels_predict_fns_list,
                        mean_f=mean_f,
                        covariance_f=covariance_f,
                        covariates_computed=covariates_computed,
                    )
                for key, value in model_based_shapley_score.items():
                    shapley_score_permutation[key][use_interval] = value

                if compute_mrmr_r2:
                    mrmr_r2_shapley_score = self.compute_mrmr_r2_based(
                        hashed_available_intervals=hashed_available_intervals,
                        available_intervals=available_intervals,
                        mapping_abscissa_interval=mapping_abscissa_interval,
                        available_intervals_computed=available_intervals_computed,
                        shapley_scores_computed=shapley_scores_computed,
                    )
                    shapley_score_permutation["mRMR_r2"][use_interval] = mrmr_r2_shapley_score
                if compute_mrmr_distance_correlation:
                    mrmr_dist_corr_shapley_score = self.compute_mrmr_dist_corr_based(
                        hashed_available_intervals=hashed_available_intervals,
                        available_intervals=available_intervals,
                        available_intervals_computed=available_intervals_computed,
                        shapley_scores_computed=shapley_scores_computed,
                        H_matrix=H_matrix,
                        dict_distance_diff_covariable=dict_distance_diff_covariable,
                        distance_target_centered=distance_target_centered,
                        dist_covariance_yy=dist_covariance_yy,
                    )

                    shapley_score_permutation["mRMR_distance_correlation"][use_interval] = mrmr_dist_corr_shapley_score
                # Store in cache the permutation at the end of the flow
                if hashed_available_intervals not in available_intervals_computed:
                    available_intervals_computed.add(hashed_available_intervals)
            # Compute the differnece of scores
            for key in shapley_score_permutation.keys():
                diff = shapley_score_permutation[key][True] - shapley_score_permutation[key][False]
                set_differences[key].append(diff)

        # Compute the mean value
        for key in mean_value.keys():
            mean_value[key] = np.mean(set_differences[key])
        return mean_value

    def plot(self, shapley_object=None):
        main_dict = self.get_main_dictionary()
        minimum_keys = main_dict.keys()
        shapley_keys = []
        if not shapley_object is None:
            print("using object")
            object_to_use = shapley_object
        else:
            object_to_use = self.shapley_values
        # If no execution has been done, throw an error
        for key in object_to_use.keys():
            if not key in minimum_keys:
                shapley_keys.append(key)
        if len(shapley_keys) == 0:
            raise RuntimeError("Please, either run `compute_shapley_value` method or include a shapley_object before plotting")
        x_val = object_to_use["middle_points"]
        print()
        n_plots = len(shapley_keys)
        fig, axs = plt.subplots(n_plots, 1, figsize=(20, 20))
        plt.subplots_adjust(hspace=0.5)
        if n_plots == 1:
            y_val = object_to_use[shapley_keys[0]]
            axs.plot(x_val, y_val)
        else:
            for i_plot, key in enumerate(shapley_keys):
                y_val = object_to_use[key]
                axs[i_plot].plot(x_val, y_val)
                axs[i_plot].set_title(key)

    def get_main_dictionary(self):
        main_dic = {
            "intervals": [],
            "middle_points": [],
        }
        return main_dic

    def create_labels(
        self,
        labels_fns,
        predict_fns,
    ):
        labels = []
        if isinstance(predict_fns, list):
            if labels_fns is None:
                total_fns = len(predict_fns)
                labels = [i for i in range(total_fns)]
            elif isinstance(labels_fns, list):
                if len(labels_fns) != len(np.unique(labels_fns)):
                    raise ValueError("`labels_fns` must contain unique values")
                if len(labels_fns) == len(predict_fns):
                    labels = [lb for lb in labels_fns]
                else:
                    raise ValueError("`labels_fns` must have the same length as `predict_fns`")
            else:
                raise ValueError("`labels_fns` must be either a list or None")
        elif callable(predict_fns):
            if labels_fns is None:
                labels = [0]
            elif isinstance(labels_fns, list):
                if len(labels_fns) > 1:
                    raise ValueError("`labels_fns` must be None, a string or a list of length 1")
                else:
                    labels = [labels_fns[0]]
            elif isinstance(labels_fns, str) or isinstance(labels_fns, int) or isinstance(labels_fns, float):
                labels = [labels_fns]
            else:
                raise ValueError("`must be either None, a string a number (or a list of length 1)`")
        return labels

    def  get_shapley_strategies(
            self,
            predict_fns,
            labels,
            compute_mrmr_r2,
            compute_mrmr_distance_correlation,
        ):
            predict_fn_list = []
            strategy = self.get_main_dictionary()
            # If it is a callable, create a list with a single elements
            if callable(predict_fns):
                strategy[labels[0]] = []
                predict_fn_list.append(predict_fns)
            # If it is a list of callables, create multiple lists
            elif isinstance(predict_fns, list):
                for predict_fn, label_fn in zip(predict_fns, labels):
                    if callable(predict_fn):
                        strategy[label_fn] = []
                        predict_fn_list.append(predict_fn)
                    else:
                        raise ValueError("All elements of `predict_fns` must be callable objects")
            # If the only computation is mrmr_r2 method, create an empty list
            else:
                if not (compute_mrmr_r2 or compute_mrmr_distance_correlation):
                    str_1 = "`predict_fns` must be either a callable or a list of callable. "
                    str_2 = "If it is None, then either `compute_mrmr_r2` or `compute_mrmr_distance_correlation` must be set to True"
                    raise ValueError(str_1 + str_2)
            if compute_mrmr_r2:
                    strategy["mRMR_r2"] = []
            if compute_mrmr_distance_correlation:
                    strategy["mRMR_distance_correlation"] = []
            return strategy, predict_fn_list

    def build_double_centering_matrix(
        self,
        n,
    ):
        ones_vector = np.full(
            shape=(n, 1),
            fill_value=1,
            dtype="int32",
        )
        identity_matrix = np.identity(n=n, dtype="float32")
        H_matrix = np.subtract(
            identity_matrix,
            np.multiply(
                1/n,
                np.matmul(ones_vector, ones_vector.T)
            )
        )
        return H_matrix

    def compute_shapley_value(
            self,
            num_permutations,
            predict_fns=None,
            labels_fns=None,
            num_intervals=None,
            intervals=None,
            seed=None,
            compute_mrmr_r2=True,
            compute_mrmr_distance_correlation=True,
        ):
        # Create labels
        labels_predict_fns_list = self.create_labels(
            labels_fns=labels_fns,
            predict_fns=predict_fns,
        )
        # Create a set of intervals: 
        #       we will treat all the intervals as [a, b), 
        #       except for the las one, which will be [a, b]
        results, predict_fn_list = self.get_shapley_strategies(
            predict_fns=predict_fns,
            labels=labels_predict_fns_list,
            compute_mrmr_r2=compute_mrmr_r2,
            compute_mrmr_distance_correlation=compute_mrmr_distance_correlation,
        )
        self.shapley_values = results.copy()
        set_intervals = self.create_set_intervals(num_intervals, intervals)
        self.print("set_intervals:\n", set_intervals)
        # Perform validations
        self.validations(num_intervals, set_intervals)
        num_intervals = set_intervals.shape[0] if num_intervals is None else num_intervals
        # Get the set of permutations
        set_permutations = self.create_permutations(
            num_intervals=num_intervals,
            num_permutations=num_permutations,
            seed=seed
        )
        total_individuals = self.X.shape[0]
        self.print("set_permutations:", set_permutations)
        # Map each abscissa point with its interval
        mapping_abscissa_interval = self.map_abscissa_interval(set_intervals)
        self.print("abscissa:", self.abscissa_points, "\n", "abscissa_interval:", mapping_abscissa_interval)
        # Compute mean value and covariance matrix
        mean_f = np.reshape(np.mean(self.X, axis=0), newshape=(-1, 1))
        covariance_f = np.cov(self.X, rowvar=False, bias=True)
        # shapley_scores_computed is used to save the scores (shapley values) with the aim to save time since we
        # avoid computing them again
        main_dict = self.get_main_dictionary()
        main_dict_keys = main_dict.keys()
        results_keys = results.keys()
        shapley_scores_computed = {k: dict() for k in results_keys if not k in main_dict_keys}
        covariates_computed = dict()
        available_intervals_computed = set()
        distance_diff_target = None
        distance_target_centered = None
        dist_covariance_yy = None
        dict_distance_diff_covariable = dict()
        H_matrix = None

        if compute_mrmr_distance_correlation:
            dict_distance_diff_covariable, distance_diff_target = self.compute_distance_difference(
                intervals=set_intervals,
                mapping_abscissa_interval=mapping_abscissa_interval,
            )
            H_matrix = self.build_double_centering_matrix(n=total_individuals)
            distance_target_centered = self.double_center_matix(
                H=H_matrix,
                D=distance_diff_target,
            )
            dist_covariance_yy = self.compute_sq_root_mean_of_products(
                X1=distance_target_centered,
                X2=distance_target_centered,
            )

        # For each interval, compute the relevance
        for i_interval in range(num_intervals):
            interval = set_intervals[i_interval]
            self.print("Computing relevance for interval:", interval, "whose index is", i_interval)
            relevance = self.compute_interval_relevance(
                set_permutations=set_permutations,
                mapping_abscissa_interval=mapping_abscissa_interval,
                mean_f=mean_f,
                covariance_f=covariance_f,
                interval_position=i_interval,
                covariates_computed=covariates_computed,
                available_intervals_computed=available_intervals_computed,
                shapley_scores_computed=shapley_scores_computed,
                compute_mrmr_r2=compute_mrmr_r2,
                compute_mrmr_distance_correlation=compute_mrmr_distance_correlation,
                predict_fn_list=predict_fn_list,
                labels_predict_fns_list=labels_predict_fns_list,
                dict_distance_diff_covariable=dict_distance_diff_covariable,
                distance_target_centered=distance_target_centered,
                dist_covariance_yy=dist_covariance_yy,
                H_matrix=H_matrix,
            )
            results["intervals"].append(interval)
            results["middle_points"].append((interval[0] + interval[1])/2)
            for key, value in relevance.items():
                results[key].append(value)
        self.shapley_values = results.copy()
        return results
