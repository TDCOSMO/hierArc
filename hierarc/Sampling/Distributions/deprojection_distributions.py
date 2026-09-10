import numpy as np
from scipy.stats import truncnorm

_SUPPORTED_DISTRIBUTIONS = ["GAUSSIAN", "GAUSSIAN_SCALED", "NONE"]
_PARAMETERIZATION = ["q_intrinsic"]


class DeprojectionDistribution(object):
    """Class to draw axisymmetric deprojection parameters (inclination/q_intrinsic) from
    hyperparameter distributions."""

    param_names = ["q_intrinsic", "q_intrinsic_sigma"]

    def __init__(
        self,
        deprojection_sampling,
        distribution_function,
        kwargs_deprojection_min,
        kwargs_deprojection_max,
        parameterization="q_intrinsic",
    ):
        """

        :param deprojection_sampling: bool, if True adds a global stellar deprojection parameter that alters the single lens
         kinematic prediction
        :param distribution_function: string, 'NONE', 'GAUSSIAN', 'GAUSSIAN_SCALED', "GAUSSIAN_TAN_RAD"
         description of the distribution function of the deprojection model parameters
        :param kwargs_deprojection_min: dictionary of bounds in the parameterization (from the interpolation)
        :param kwargs_deprojection_max: dictionary of bounds in the parameterization (from the interpolation)
        :param parameterization: model of parameterization (currently for constant deprojection), ["beta" or "TAN_RAD"]
        """
        self._deprojection_sampling = deprojection_sampling
        if distribution_function not in _SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                "Anisotropy distribution function %s not supported. Chose among %s."
                % (distribution_function, _SUPPORTED_DISTRIBUTIONS)
            )
        if parameterization not in _PARAMETERIZATION:
            raise ValueError(
                "Deprojection parameterization %s not supported. Chose among %s."
                % (parameterization, _PARAMETERIZATION)
            )
        self._distribution_function = distribution_function
        self._parametrization = parameterization
        if kwargs_deprojection_min is None:
            kwargs_deprojection_min = {}
        if kwargs_deprojection_max is None:
            kwargs_deprojection_max = {}
        self._kwargs_min = kwargs_deprojection_min
        self._kwargs_max = kwargs_deprojection_max
        self._q_min, self._q_max = self._kwargs_min.get(
            "q_intrinsic", 0.0
        ), self._kwargs_max.get("q_intrinsic", 1.0)

    def draw_deprojection(self, q_intrinsic=None, q_intrinsic_sigma=0):
        """Draw from the population distribution, restricted to the interpolated range.

        The bounds are those of *this lens's* q_intrinsic grid, which the pre-processing
        caps at the observed q_mass of the lens. The sampled q_intrinsic, on the other
        hand, is the mean of the population, and a population mean is not required to lie
        inside the support of every member: what has to stay inside the grid is the value
        the kinematics scaling J is interpolated at, i.e. the draw.

        So the mean is only required to be inside the range when it *is* the value handed
        to the interpolator - no distribution, or a distribution of zero width. With a
        distribution of finite width the draw is taken from the normal truncated to the
        range, which is the same law rejection sampling defines and the same one the
        deterministic rule integrates (QuadratureMarginalisation, whose weights
        Util.quadrature_util.truncated_normal_panels normalises over the same bounds).
        Sampling it by its inverse CDF rather than by rejection also terminates when the
        mean sits well outside the range, where rejection would recurse indefinitely.

        :param q_intrinsic: mean of the distribution
        :param q_intrinsic_sigma: std of the distribution
        :return: random draw from the distribution
        """
        kwargs_return = {}
        q = q_intrinsic
        if not self._deprojection_sampling:
            if q_intrinsic is not None:
                kwargs_return["q_intrinsic"] = q
            return kwargs_return
        if self._distribution_function in ["GAUSSIAN", "GAUSSIAN_SCALED"]:
            if self._distribution_function in ["GAUSSIAN"]:
                sigma = q_intrinsic_sigma
            else:
                sigma = q_intrinsic_sigma * q_intrinsic
        else:
            sigma = 0
        if sigma <= 0:
            # the mean itself reaches the interpolator
            if q <= self._q_min or q > self._q_max:
                raise ValueError(
                    "deprojection parameter with %s is out of bounds of the interpolated range [%s, %s]!"
                    % (q, self._q_min, self._q_max)
                )
            kwargs_return["q_intrinsic"] = q
            return kwargs_return
        a = (self._q_min - q_intrinsic) / sigma
        b = (self._q_max - q_intrinsic) / sigma
        kwargs_return["q_intrinsic"] = float(
            truncnorm.rvs(a, b, loc=q_intrinsic, scale=sigma)
        )
        return kwargs_return

    def get_deprojection_sampling_params(self, kwargs_kin):
        """Gets the deprojection distribution parameters from a larger dictionary.

        :param kwargs_kin: Dictionary of kinematic parameters
        :return: Dictionary of deprojection parameters
        """
        deprojection_params = {}
        if "q_intrinsic" in kwargs_kin:
            deprojection_params["q_intrinsic"] = kwargs_kin["q_intrinsic"]
        if self._deprojection_sampling and ("q_intrinsic_sigma" in kwargs_kin):
            deprojection_params["q_intrinsic_sigma"] = kwargs_kin["q_intrinsic_sigma"]
        return deprojection_params
