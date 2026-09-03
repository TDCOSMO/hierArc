import numpy as np
import math
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity


class DdtHistLikelihood(object):
    """Evaluates the likelihood of a time-delay distance ddt (in Mpc) against the model
    predictions, using a loglikelihood sampled from a Kernel Density Estimator. The KDE
    is constructed using a binned version of the full samples. Greatly improves speed at
    the cost of a (tiny) loss in precision.

    .. warning::

        you should adjust bandwidth and nbins_hist to the spacing and
        size of your samples chain!

    original source:
    https://github.com/shsuyu/H0LiCOW-public/blob/master/H0_inference_code/lensutils.py
    credits to Martin Millon, Aymeric Galan
    """

    def __init__(
        self,
        z_lens,
        z_source,
        ddt_samples,
        ddt_weights=None,
        nbins_hist=200,
        normalized=False,
        binning_method=None,
    ):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param ddt_samples: numpy array of Ddt values
        :param ddt_weights: optional weights for the samples in Ddt
        :param nbins_hist: number of bins in the histogram
        :param normalized: bool, if True, returns the normalized likelihood,
         if False, separates the constant prefactor (in case of a Gaussian
         1/(sigma sqrt(2 pi))) to compute the reduced chi2 statistics
        :param binning_method: method used to calculate the bandwidth. "scott",
         "silverman", and a scalar constant (KDE bandwidth) are supported.
         (See the scipy.stats.gaussian_kde documentation for details.)

        """
        if binning_method is None:
            hist = np.histogram(ddt_samples, bins=nbins_hist, weights=ddt_weights)
            vals = hist[0]
            bins = [(h + hist[1][i + 1]) / 2.0 for i, h in enumerate(hist[1][:-1])]
            # ignore potential zero weights, sklearn does not like them
            kde_bins = [b for v, b in zip(vals, bins) if v > 0]
            kde_weights = [v for v in vals if v > 0]
            self._kde = gaussian_kde(dataset=kde_bins, weights=kde_weights[:])
        else:
            self._kde = gaussian_kde(
                dataset=ddt_samples, bw_method=binning_method, weights=ddt_weights
            )
        self.num_data = 1
        self._sigma = np.std(ddt_samples)
        self._norm_factor = 0
        if normalized is False:
            self._norm_factor = np.log(1.0 / self._sigma / np.sqrt(2 * np.pi))
        self._ddt_mean = np.average(ddt_samples, weights=ddt_weights)
        variance = np.average((ddt_samples - self._ddt_mean) ** 2, weights=ddt_weights)
        self._ddt_sigma = math.sqrt(variance)

    def log_likelihood(self, ddt, dd=None):
        """
        Note: kinematics + imaging data can constrain Ds/Dds. The input of Ddt,
        Dd is transformed here to match Ds/Dds

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :return: log likelihood given the single lens analysis

        """
        return self._kde.logpdf(ddt) - self._norm_factor

    def ddt_measurement(self):
        """

        :return: mean, 1-sigma of the ddt inference/model measurement

        """
        return self._ddt_mean, self._ddt_sigma


class DdtHistKDELikelihood(object):
    """Evaluates the likelihood of a time-delay distance ddt (in Mpc) against the model
    predictions, using a loglikelihood sampled from a Kernel Density Estimator. the KDE
    is constructed using a binned version of the full samples. Greatly improves speed at
    the cost of a (tiny) loss in precision.

    __warning:: you should adjust bandwidth and nbins_hist to the spacing and size of your samples chain!

    original source: https://github.com/shsuyu/H0LiCOW-public/blob/master/H0_inference_code/lensutils.py
    credits to Martin Millon, Aymeric Galan
    """

    def __init__(
        self,
        z_lens,
        z_source,
        ddt_samples,
        kde_kernel="gaussian",
        ddt_weights=None,
        bandwidth=20,
        nbins_hist=200,
        normalized=False,
    ):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param ddt_samples: numpy array of Ddt values
        :param ddt_weights: optional weights for the samples in Ddt
        :param kde_kernel: string of KDE kernel type
        :param bandwidth: bandwith of kernel
        :param nbins_hist: number of bins in the histogram
        :param normalized: bool, if True, returns the normalized likelihood, if False, separates the constant prefactor
         (in case of a Gaussian 1/(sigma sqrt(2 pi)) ) to compute the reduced chi2 statistics
        """

        vals, bin_edges = np.histogram(
            ddt_samples, bins=nbins_hist, weights=ddt_weights, density=True
        )
        bins = [(h + bin_edges[i + 1]) / 2.0 for i, h in enumerate(bin_edges[:-1])]

        # ignore potential zero weights, sklearn does not like them
        kde_bins = [(b,) for v, b in zip(vals, bins) if v > 0]
        kde_weights = [v for v in vals if v > 0]

        kde = KernelDensity(kernel=kde_kernel, bandwidth=bandwidth).fit(
            kde_bins, sample_weight=kde_weights
        )
        self._score = kde.score
        # Fast path for the (default) Gaussian kernel. scikit-learn's score() spends
        # most of its time validating its arguments, which is wasted here: the
        # hierarchical likelihood calls it once per population draw per lens, always on
        # a single scalar Ddt, against a fixed mixture of at most nbins_hist
        # components. Evaluating that mixture directly is ~8x faster and agrees with
        # score() to ~1e-9. Any other kernel keeps using scikit-learn.
        self._kde_centers, self._kde_log_weights, self._kde_norm = None, None, None
        if kde_kernel == "gaussian":
            weights = np.asarray(kde_weights, dtype=float)
            self._kde_centers = np.asarray(kde_bins, dtype=float).ravel()
            self._kde_log_weights = np.log(weights) - np.log(np.sum(weights))
            self._kde_bandwidth = float(bandwidth)
            self._kde_norm = -0.5 * np.log(2 * np.pi) - np.log(self._kde_bandwidth)
        self.num_data = 1
        self._sigma = np.std(ddt_samples)
        self._norm_factor = 0
        if normalized is False:
            self._norm_factor = np.log(1.0 / self._sigma / np.sqrt(2 * np.pi))
        self._ddt_mean = np.average(ddt_samples, weights=ddt_weights)
        variance = np.average((ddt_samples - self._ddt_mean) ** 2, weights=ddt_weights)
        self._ddt_sigma = math.sqrt(variance)

    def log_likelihood(self, ddt, dd=None):
        """
        Note: kinematics + imaging data can constrain Ds/Dds. The input of Ddt, Dd is transformed here to match Ds/Dds

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :return: log likelihood given the single lens analysis
        """
        if self._kde_centers is None:
            return self._score(np.array(ddt).reshape(1, -1)) - self._norm_factor
        # log of the Gaussian mixture, with the maximum factored out so that a Ddt far
        # from the support underflows to -inf rather than to a nan
        exponent = (
            self._kde_log_weights
            - 0.5 * ((ddt - self._kde_centers) / self._kde_bandwidth) ** 2
        )
        exponent_max = np.max(exponent)
        log_density = (
            exponent_max
            + np.log(np.sum(np.exp(exponent - exponent_max)))
            + self._kde_norm
        )
        return log_density - self._norm_factor

    def log_likelihood_vector(self, ddt):
        """Log likelihood for a whole array of Ddt at once.

        Only available for the Gaussian kernel, where the KDE is a fixed mixture that
        can be evaluated in one broadcast. Used by the deterministic marginalisation,
        which needs the Ddt term on a few hundred quadrature nodes at once.

        :param ddt: array of time-delay distances
        :return: array of log likelihoods
        :raises NotImplementedError: for a non-Gaussian kernel
        """
        if self._kde_centers is None:
            raise NotImplementedError(
                "vectorised evaluation is only implemented for the Gaussian kernel"
            )
        ddt = np.atleast_1d(np.asarray(ddt, dtype=float))
        exponent = (
            self._kde_log_weights[None, :]
            - 0.5
            * ((ddt[:, None] - self._kde_centers[None, :]) / self._kde_bandwidth) ** 2
        )
        exponent_max = np.max(exponent, axis=1)
        log_density = (
            exponent_max
            + np.log(np.sum(np.exp(exponent - exponent_max[:, None]), axis=1))
            + self._kde_norm
        )
        return log_density - self._norm_factor

    def ddt_measurement(self):
        """

        :return: mean, 1-sigma of the ddt inference/model measurement
        """
        return self._ddt_mean, self._ddt_sigma
