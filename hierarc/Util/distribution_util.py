import numpy as np
from scipy.interpolate import interp1d


class PDFSampling(object):
    """Class for approximations with a given pdf sample."""

    def __init__(self, bin_edges, pdf_array):
        """

        :param bin_edges: bin edges of PDF values
        :param pdf_array: pdf array of given bins (len(bin_edges)-1)
        """
        assert len(bin_edges) == len(pdf_array) + 1
        # kept so that a deterministic rule can be built for the same distribution
        self.bin_edges = np.asarray(bin_edges, dtype=float)
        self.pdf_array = np.asarray(pdf_array, dtype=float)
        self._cdf_array, self._cdf_func, self._cdf_inv_func = approx_cdf_1d(
            bin_edges, pdf_array
        )

    def draw(self, n=1):
        """

        :return:
        """
        p = np.random.uniform(0, 1, n)
        return self._cdf_inv_func(p)

    @property
    def draw_one(self):
        """

        :return:
        """
        return self.draw(n=1)


class DistributionSampling(object):
    """Class to sample from discrete samples (can be multi-dimensional)"""

    def __init__(self, distributions):
        """

        :param distributions: list of distributions
        """
        self._num_dist = len(distributions)
        self._distributions = distributions

    @property
    def draw_one(self):
        """Randomly selects single point in distribution.

        :return:
        """
        index = np.random.randint(low=0, high=self._num_dist)
        return self._distributions[index]


def approx_cdf_1d(bin_edges, pdf_array):
    """

    :param bin_edges: bin edges of PDF values
    :param pdf_array: pdf array of given bins (len(bin_edges)-1)
    :return: cdf, interp1d function of cdf, inverse interpolation function
    """
    assert len(bin_edges) == len(pdf_array) + 1
    norm_pdf = pdf_array / np.sum(pdf_array)
    cdf_array = np.zeros_like(bin_edges)
    cdf_array[0] = 0
    for i in range(0, len(norm_pdf)):
        cdf_array[i + 1] = cdf_array[i] + norm_pdf[i]
    cdf_func = interp1d(bin_edges, cdf_array)
    cdf_inv_func = interp1d(cdf_array, bin_edges)
    return cdf_array, cdf_func, cdf_inv_func
