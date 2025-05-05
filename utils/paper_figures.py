from matplotlib.collections import LineCollection
import numpy as np
import warnings
import matplotlib.pyplot as plt
import numpy as np


class PaperFiguresTools:
    def __init__(
        self,
        domain_range,
        abscissa_points,
        X,
    ):
        self.domain_range = domain_range
        self.abscissa_points = abscissa_points
        self.X = X

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
            raise NotImplemented("`Please, provide num_intervals while we implement this part`")
        return intervals

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
        if invertibility < 1e-100:
            inv_covariance_available_abscissa = np.linalg.pinv(covariance_available_abscissa)
        else:    
            inv_covariance_available_abscissa = np.linalg.inv(covariance_available_abscissa)
        matrix_mult = np.matmul(covariance_mix, inv_covariance_available_abscissa)
        conditional_expectation_fn = self.conditional_expectation(
            mean_1=mean_non_available_abscissa,
            mean_2=mean_available_abscissa,
            matrix_mult=matrix_mult
        )
        total_individuals = self.X.shape[0]
        for i in range(total_individuals):
            X_i_available_abscissa = np.reshape(
                self.X[i][position_available_abscissa],
                newshape=(-1, 1)
            )
            conditional_expectation_i = np.ravel(conditional_expectation_fn(X_i_available_abscissa))
            recomputed_covariate[i][position_non_available_abscissa] = conditional_expectation_i
        return recomputed_covariate

def colored_line(
    x,
    y,
    c,
    ax,
    linestyle,
    **lc_kwargs
):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)
    print(segments.shape)
    lc = LineCollection(segments, linestyle=linestyle, colors=c, **default_kwargs)
    #lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)
