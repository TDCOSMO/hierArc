__author__ = "sibirrer", "ajshajib"

from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator
import numpy as np


class KinScalingParamManager(object):
    """Class to handle the sorting of parameters in the kinematics scaling."""

    def __init__(self, j_kin_scaling_param_name_list):
        """

        :param j_kin_scaling_param_name_list: list of strings for the parameters as they are interpolated in the same
         order as j_kin_scaling_grid
        """
        if j_kin_scaling_param_name_list is None:
            self._param_list = []
        else:
            self._param_list = j_kin_scaling_param_name_list
        self._num_param = len(self._param_list)

    @property
    def kin_scaling_param_names(self):
        """Names of the interpolated parameters, in grid order.

        :return: list of str
        """
        return list(self._param_list)

    @property
    def num_scaling_dim(self):
        """Number of parameter dimensions for kinematic scaling.

        :return: number of scaling dimensions
        :rtype: int
        """
        return self._num_param

    def kwargs2param_array(self, kwargs):
        """Converts dictionary to sorted array in same order as interpolation grid.

        :param kwargs: dictionary of all model components, must include the one that are
            interpolated
        :return: sorted list of parameters to interpolate
        """
        param_array = []
        for param in self._param_list:
            if param not in kwargs:
                raise ValueError(
                    "key %s not in parameters and hence kinematic scaling not possible"
                    % param
                )
            param_array.append(kwargs.get(param))
        return param_array

    def param_array2kwargs(self, param_array):
        """Inverse function of kwargs2param_array for a given param_array returns the
        dictionary split in anisotropy and lens models.

        :param param_array:
        :return: kwargs_anisotropy, kwargs_lens, kwargs_deprojection
        """
        kwargs_anisotropy, kwargs_lens, kwargs_deprojection = {}, {}, {}
        for i, param in enumerate(self._param_list):
            if param in ["gamma_in", "gamma_pl", "log_m2l"]:
                kwargs_lens[param] = param_array[i]
            elif param in ["q_intrinsic"]:
                kwargs_deprojection[param] = param_array[i]
            else:
                kwargs_anisotropy[param] = param_array[i]
        return kwargs_anisotropy, kwargs_lens, kwargs_deprojection


class ParameterScalingSingleMeasurement(object):
    """Class to manage anisotropy scaling for single slit observation."""

    def __init__(self, param_grid_axes, j_kin_scaling_grid):
        """

        :param param_grid_axes: list of arrays of interpolated parameter values
        :param j_kin_scaling_grid: array with the scaling of J() for single measurement bin in same dimensions as the
         param_arrays
        """
        self._evalute_scaling = False
        # check if param arrays is 1d list or 2d list
        if param_grid_axes is not None and j_kin_scaling_grid is not None:
            if isinstance(param_grid_axes, list):
                self._dim_scaling = len(param_grid_axes)
            else:
                self._dim_scaling = 1
                param_grid_axes = [param_grid_axes]

            if self._dim_scaling == 1:
                self._f_ani = interp1d(
                    param_grid_axes[0],
                    j_kin_scaling_grid,
                    kind="linear",
                    fill_value="extrapolate",
                )
            elif self._dim_scaling == 2:
                self._f_ani = r = RectBivariateSpline(
                    param_grid_axes[0],
                    param_grid_axes[1],
                    j_kin_scaling_grid,
                    kx=min(len(param_grid_axes[0]) - 1, 3),
                    ky=min(len(param_grid_axes[1]) - 1, 3),
                )
                # self._f_ani = interp2d(
                #    param_grid_axes[0], param_grid_axes[1], j_kin_scaling_grid.T
                # )
            else:
                self._f_ani = RegularGridInterpolator(
                    tuple(param_grid_axes),
                    j_kin_scaling_grid,
                )
            self._evalute_scaling = True

    def j_scaling(self, param_array):
        """

        :param param_array: sorted list of parameters for the interpolation function
        :return: scaling J(a_ani) for single slit
        """
        if self._evalute_scaling is not True or len(param_array) == 0:
            return 1
        if self._dim_scaling == 1:
            return self._f_ani(param_array[0])
        elif self._dim_scaling == 2:
            return self._f_ani(param_array[0], param_array[1])[0].T[0]
        else:
            return self._f_ani(param_array)[0]


class KinScaling(KinScalingParamManager):
    """Class to manage model parameter and anisotropy scalings for IFU data."""

    def __init__(
        self,
        j_kin_scaling_param_axes=None,
        j_kin_scaling_grid_list=None,
        j_kin_scaling_param_name_list=None,
    ):
        """

        :param j_kin_scaling_param_axes: array of parameter values for each axes of j_kin_scaling_grid
        :param j_kin_scaling_grid_list: list of array with the scalings of J() for each IFU
        :param j_kin_scaling_param_name_list: list of strings for the parameters as they are interpolated in the same
         order as j_kin_scaling_grid
        """

        self._param_arrays = j_kin_scaling_param_axes
        if (
            not isinstance(j_kin_scaling_param_axes, list)
            and j_kin_scaling_param_name_list is not None
        ):
            self._param_arrays = [j_kin_scaling_param_axes]
        self._evaluate_scaling = False
        self._is_log_m2l_population_level = False
        self._f_ani_stacked = None
        if (
            j_kin_scaling_param_axes is not None
            and j_kin_scaling_grid_list is not None
            and j_kin_scaling_param_name_list is not None
        ):
            self._evaluate_scaling = True
            self._j_scaling_ifu = []
            self._f_ani_list = []
            for scaling_grid in j_kin_scaling_grid_list:
                self._j_scaling_ifu.append(
                    ParameterScalingSingleMeasurement(
                        j_kin_scaling_param_axes, scaling_grid
                    )
                )
            self._f_ani_stacked = self._stack_measurements(
                j_kin_scaling_param_axes, j_kin_scaling_grid_list
            )

        if isinstance(j_kin_scaling_param_axes, list):
            self._dim_scaling = len(j_kin_scaling_param_axes)
        else:
            self._dim_scaling = 1
        KinScalingParamManager.__init__(
            self, j_kin_scaling_param_name_list=j_kin_scaling_param_name_list
        )

    @staticmethod
    def _stack_measurements(param_grid_axes, j_kin_scaling_grid_list):
        """Build a single interpolator evaluating all measurement bins at once.

        Every bin of a lens is tabulated on the same parameter grid and only the values
        differ, so the bins can be stacked into one extra (trailing) dimension of the
        value array and interpolated by a single call. This is purely a performance
        rewrite of the per-bin loop in :func:`kin_scaling`: the interpolation itself is
        unchanged, and both paths return the same numbers.

        It matters because the cost of these interpolators is dominated by the
        per-call Python overhead (argument validation, index search), not by the
        arithmetic: one call for all bins is ~n_bins times cheaper than one call per
        bin. The hierarchical likelihood evaluates this once per population draw per
        lens, so the loop is the hot path of the whole inference.

        Only the 1d (``interp1d``) and >=3d (``RegularGridInterpolator``) cases are
        stacked. The 2d case uses a ``RectBivariateSpline``, which has no
        vector-valued form, and keeps the per-bin loop.

        :param param_grid_axes: list of arrays of interpolated parameter values
        :param j_kin_scaling_grid_list: list of J() grids, one per measurement bin
        :return: interpolator returning one value per bin, or None if the
            configuration cannot be stacked
        """
        if not isinstance(param_grid_axes, list):
            param_grid_axes = [param_grid_axes]
        dim_scaling = len(param_grid_axes)
        if dim_scaling == 2:
            return None
        grids = [np.asarray(grid, dtype=float) for grid in j_kin_scaling_grid_list]
        if len(grids) == 0:
            return None
        if any(grid.shape != grids[0].shape for grid in grids):
            return None

        if dim_scaling == 1:
            # values of shape (n_bins, n_x), interpolated along the last axis so that a
            # scalar argument returns one value per bin
            return interp1d(
                param_grid_axes[0],
                np.stack(grids, axis=0),
                kind="linear",
                fill_value="extrapolate",
                axis=-1,
            )
        # values of shape (*grid_shape, n_bins): RegularGridInterpolator interpolates
        # over the leading axes and carries the trailing one through untouched
        return RegularGridInterpolator(tuple(param_grid_axes), np.stack(grids, axis=-1))

    @property
    def kin_scaling_axes(self):
        """Interpolation axes of the kinematics scaling, in the same order as the names.

        :return: list of arrays, or None if no scaling is interpolated
        """
        return self._param_arrays

    def param_bounds_interpol(self):
        """Minimum and maximum bounds of parameters that are being used to call
        interpolation function.

        :return: dictionaries of minimum and maximum bounds
        """
        kwargs_min, kwargs_max = {}, {}
        if self._evaluate_scaling is True:
            for i, key in enumerate(self._param_list):
                kwargs_min[key] = min(self._param_arrays[i])
                kwargs_max[key] = max(self._param_arrays[i])
        return kwargs_min, kwargs_max

    def kin_scaling(self, kwargs_param):
        """

        :param kwargs_param: dictionary of parameters for scaling the kinematics
        :return: scaling J(a_ani) for the IFU's
        """
        if kwargs_param is None:
            return np.ones(self._dim_scaling)
        param_array = self.kwargs2param_array(kwargs_param)
        if self._evaluate_scaling is not True or len(param_array) == 0:
            return np.ones(self._dim_scaling)
        if self._f_ani_stacked is not None:
            # all bins in one interpolator call (see _stack_measurements)
            if self._dim_scaling == 1:
                return self._f_ani_stacked(param_array[0])
            return self._f_ani_stacked(param_array)[0]
        scaling_list = []
        for scaling_class in self._j_scaling_ifu:
            scaling = scaling_class.j_scaling(param_array)
            scaling_list.append(scaling)
        return np.array(scaling_list)
