__author__ = "sibirrer"

from lenstronomy.Util import constants as const
from scipy.linalg import solve_triangular, cholesky, eigh
import numpy as np


class KinLikelihood(object):
    """Likelihood to deal with IFU kinematics constraints with covariances in both the
    model and measured velocity dispersion."""

    def __init__(
        self,
        z_lens,
        z_source,
        sigma_v_measurement,
        j_model,
        error_cov_measurement,
        error_cov_j_sqrt,
        normalized=True,
        sigma_sys_error_include=False,
    ):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param sigma_v_measurement: numpy array, velocity dispersion measured
        :param j_model: numpy array of the predicted dimensionless dispersion on the IFU's
        :param error_cov_measurement: covariance matrix of the measured velocity dispersions in the IFU's
        :param error_cov_j_sqrt: covariance matrix of sqrt(J) of the model predicted dimensionless dispersion on the IFU's
        :param normalized: bool, if True, returns the normalized likelihood, if False, separates the constant prefactor
         (in case of a Gaussian 1/(sigma sqrt(2 pi)) ) to compute the reduced chi2 statistics
        :param sigma_sys_error_include: bool, if True will include a systematic error in the velocity dispersion
         measurement (if sampled from), otherwise this sampled value is ignored.
        """
        self._z_lens = z_lens
        self._j_model = np.array(j_model, dtype=float)
        self._sigma_v_measured = np.array(sigma_v_measurement, dtype=float)
        self._error_cov_measurement = np.array(error_cov_measurement, dtype=float)
        self._error_cov_j_sqrt = np.array(error_cov_j_sqrt, dtype=float)
        self.num_data = len(j_model)
        self._normalized = normalized
        self._sigma_sys_error_include = sigma_sys_error_include
        # cached Cholesky of the measurement covariance, which never changes
        self._chol_meas, self._log_det_meas = None, None

    def log_likelihood(
        self,
        ddt,
        dd,
        kin_scaling=None,
        sigma_v_sys_error=None,
        sigma_v_sys_offset=None,
    ):
        """
        Note: kinematics + imaging data can constrain Ds/Dds. The input of Ddt, Dd is transformed here to match Ds/Dds

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param kin_scaling: array of size of the velocity dispersion measurement or None, scaling of the predicted
         dimensionless quantity J (proportional to sigma_v^2) of the anisotropy model in the sampling relative to the
         anisotropy model used to derive the prediction and covariance matrix in the init of this class.
        :param sigma_v_sys_error: float (optional) added error on the velocity dispersion measurement in quadrature
        :param sigma_v_sys_offset: float (optional) for a fractional systematic offset in the kinematic measurement
         such that sigma_v = sigma_v_measured * (1 + sigma_v_sys_offset)
        :return: log likelihood given the single lens analysis
        """
        ds_dds = np.maximum(ddt / dd / (1 + self._z_lens), 0)
        if kin_scaling is None:
            scaling_ifu = 1
        else:
            scaling_ifu = kin_scaling
        sigma_v_predict = self.sigma_v_model(ds_dds, scaling_ifu)
        delta = self.sigma_v_measurement_mean(sigma_v_sys_offset) - sigma_v_predict
        cov_error = self.cov_error_measurement(
            sigma_v_sys_error
        ) + self.cov_error_model(ds_dds, scaling_ifu)
        try:
            # this is faster and more stable than the matrix inversion
            cov_scale = np.linalg.cholesky(cov_error)
            y = solve_triangular(cov_scale, delta, lower=True)
            lnlikelihood = -np.dot(y, y) / 2.0
        except np.linalg.LinAlgError:
            # cov_error is not positive definite
            return -np.inf
        # lnlikelihood = -delta.dot(cov_error_inv.dot(delta)) / 2.0
        if self._normalized is True:
            sign_det, lndet = np.linalg.slogdet(cov_error)
            if sign_det < 0:
                raise ValueError(
                    "error covariance matrix needs to be positive definite"
                )
            lnlikelihood -= 1 / 2.0 * (self.num_data * np.log(2 * np.pi) + lndet)
        return lnlikelihood

    def pencil_setup(self, kin_scaling=None, sigma_v_sys_error=None):
        """Pre-diagonalise the covariance so that Ds/Dds becomes a cheap scalar sweep.

        With the anisotropy/deprojection draw fixed, the covariance of this lens is an
        affine family in t = Ds/Dds,

            Sigma(t) = C + t M,   M = error_cov_j_sqrt * outer(sqrt(J), sqrt(J)) * c^2,

        because multiplying elementwise by an outer product is a diagonal congruence.
        The residual is affine in sqrt(t) for the same reason,

            delta(t) = sigma_v_measured - sqrt(t) * s,   s = sqrt(j_model * J) * c.

        Factoring C = L L^T and diagonalising L^-1 M L^-T = Q Lambda Q^T puts *both*
        matrices in one basis, in which Sigma(t) is diagonal for every t. The quadratic
        form and the log-determinant then follow in O(n) per value of t rather than an
        O(n^3) Cholesky each. Because C is the measurement covariance it does not depend
        on the model at all, so its factor is computed once and cached.

        :param kin_scaling: array of the kinematics scaling J for each measurement bin
        :param sigma_v_sys_error: optional systematic velocity dispersion error,
            added in quadrature to the measurement covariance
        :return: dictionary with 'lambda', 'p', 'r' and 'log_det_c', consumed by
            :func:`log_likelihood_pencil`
        """
        if kin_scaling is None:
            scaling = np.ones(self.num_data)
        else:
            scaling = np.atleast_1d(np.asarray(kin_scaling, dtype=float))
        cov_meas = self.cov_error_measurement(sigma_v_sys_error)
        if sigma_v_sys_error is None or not self._sigma_sys_error_include:
            if self._chol_meas is None:
                self._chol_meas = cholesky(cov_meas, lower=True)
                self._log_det_meas = 2.0 * np.sum(np.log(np.diag(self._chol_meas)))
            chol, log_det_c = self._chol_meas, self._log_det_meas
        else:
            chol = cholesky(cov_meas, lower=True)
            log_det_c = 2.0 * np.sum(np.log(np.diag(chol)))

        root = np.sqrt(np.maximum(scaling, 0.0))
        m_matrix = self._error_cov_j_sqrt * np.outer(root, root) * (const.c / 1000) ** 2
        a_matrix = solve_triangular(chol, m_matrix, lower=True)
        a_matrix = solve_triangular(chol, a_matrix.T, lower=True).T
        eigenvalues, q_matrix = eigh(0.5 * (a_matrix + a_matrix.T))
        s_vector = np.sqrt(np.maximum(self._j_model * scaling, 0.0)) * const.c / 1000
        measured = solve_triangular(chol, self._sigma_v_measured, lower=True)
        return {
            "lambda": eigenvalues,
            "p": q_matrix.T @ measured,
            "r": q_matrix.T @ solve_triangular(chol, s_vector, lower=True),
            "log_det_c": log_det_c,
        }

    def log_likelihood_pencil(self, ds_dds, pencil):
        """Log likelihood for many values of Ds/Dds at the cost of O(n) each.

        :param ds_dds: scalar or array of Ds/Dds values
        :param pencil: output of :func:`pencil_setup` for the same kinematics scaling
        :return: array of log likelihoods, same shape as ``ds_dds``
        """
        t = np.atleast_1d(np.asarray(ds_dds, dtype=float))
        t = np.maximum(t, 0.0)
        denominator = 1.0 + t[:, None] * pencil["lambda"][None, :]
        residual = pencil["p"][None, :] - np.sqrt(t)[:, None] * pencil["r"][None, :]
        numerator = residual**2
        log_like = -0.5 * np.sum(numerator / denominator, axis=1)
        if self._normalized is True:
            log_like -= 0.5 * (
                self.num_data * np.log(2 * np.pi)
                + pencil["log_det_c"]
                + np.sum(np.log(denominator), axis=1)
            )
        return log_like

    def sigma_v_measurement_mean(self, sigma_v_sys_offset=None):
        """

        :param sigma_v_sys_offset: float (optional) for a fractional systematic offset in the kinematic measurement
         such that sigma_v = sigma_v_measured * (1 + sigma_v_sys_offset)
        :return: corrected measured velocity dispersion
        """
        if sigma_v_sys_offset is None:
            return self._sigma_v_measured
        else:
            return self._sigma_v_measured * (1 + sigma_v_sys_offset)

    def sigma_v_model(self, ds_dds, kin_scaling=1):
        """Model predicted velocity dispersion for the IFU's.

        :param ds_dds: Ds/Dds
        :param kin_scaling: scaling of the anisotropy affecting sigma_v^2
        :return: array of predicted velocity dispersions
        """
        sigma_v_predict = np.sqrt(self._j_model * ds_dds * kin_scaling) * const.c / 1000
        return sigma_v_predict

    def cov_error_model(self, ds_dds, kin_scaling=1):
        """

        :param ds_dds: Ds/Dds
        :param kin_scaling: scaling of the anisotropy affecting sigma_v^2
        :return: covariance matrix of the error in the predicted model (from mass model uncertainties)
        """
        scaling_matix = np.outer(np.sqrt(kin_scaling), np.sqrt(kin_scaling))
        return self._error_cov_j_sqrt * scaling_matix * ds_dds * (const.c / 1000) ** 2

    def cov_error_measurement(self, sigma_v_sys_error=None):
        """

        :param sigma_v_sys_error: float (optional) added error on the velocity dispersion measurement in quadrature
        :return: error covariance matrix of the velocity dispersion measurements
        """
        if self._sigma_sys_error_include and sigma_v_sys_error is not None:
            return self._error_cov_measurement + np.outer(
                self._sigma_v_measured * sigma_v_sys_error,
                self._sigma_v_measured * sigma_v_sys_error,
            )
        else:
            return self._error_cov_measurement

    def sigma_v_prediction(self, ddt, dd, kin_scaling=1):
        """Model prediction mean velocity dispersion vector and model prediction
        covariance matrix.

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param kin_scaling: array of size of the velocity dispersion measurement or
            None, scaling of the predicted dimensionless quantity J (proportional to
            sigma_v^2) of the anisotropy model in the sampling relative to the
            anisotropy model used to derive the prediction and covariance matrix in the
            init of this class.
        :return: model prediction mean velocity dispersion vector and model prediction
            covariance matrix
        """
        ds_dds = np.maximum(ddt / dd / (1 + self._z_lens), 0)
        sigma_v_predict = self.sigma_v_model(ds_dds, kin_scaling)
        cov_error_predict = self.cov_error_model(ds_dds, kin_scaling)
        return sigma_v_predict, cov_error_predict

    def sigma_v_measurement(self, sigma_v_sys_error=None, sigma_v_sys_offset=None):
        """

        :param sigma_v_sys_error: float (optional) added error on the velocity dispersion measurement in quadrature
        :param sigma_v_sys_offset: float (optional) for a fractional systematic offset in the kinematic measurement
         such that sigma_v = sigma_v_measured * (1 + sigma_v_sys_offset)
        :return: measurement mean (vector), measurement covariance matrix
        """
        return self.sigma_v_measurement_mean(
            sigma_v_sys_offset
        ), self.cov_error_measurement(sigma_v_sys_error)
