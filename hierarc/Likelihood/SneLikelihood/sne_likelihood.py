import numpy as np

from hierarc.Likelihood.SneLikelihood.sne_likelihood_from_file import (
    SneLikelihoodFromFile,
)
from hierarc.Likelihood.SneLikelihood.sne_likelihood_custom import CustomSneLikelihood
from hierarc.Likelihood.SneLikelihood.sne_pantheon_plus import PantheonPlusData


class SneLikelihood(object):
    """Supernovae likelihood This class supports custom likelihoods as well as
    likelihoods from the Pantheon sample from file."""

    def __init__(self, sample_name="CUSTOM", **kwargs_sne_likelihood):
        """

        :param sample_name: string, either 'CUSTOM', 'PantheonPlus', 'DES5YR', 'DES5YR_Dovekie' or a specific
            name supported by SneLikelihoodFromFile() class
        :param kwargs_sne_likelihood: keyword arguments to initiate likelihood class
        """
        if sample_name == "CUSTOM":
            self._likelihood = CustomSneLikelihood(**kwargs_sne_likelihood)
        elif sample_name == "PantheonPlus":
            from hierarc.Likelihood.SneLikelihood.sne_pantheon_plus import (
                PantheonPlusData,
            )

            data = PantheonPlusData()
            mag_mean = data.m_obs
            cov_mag = data.cov_mag_b
            zhel = data.zHEL
            zcmb = data.zCMB
            self._likelihood = CustomSneLikelihood(
                mag_mean, cov_mag, zhel, zcmb, no_intrinsic_scatter=True
            )
        elif sample_name == "DES5YR":
            from hierarc.Likelihood.SneLikelihood.des_5yr import DES5YRData

            data = DES5YRData()
            mag_mean = data.mu_obs
            cov_mag = data.cov_mag_b
            zhel = data.zHEL
            zcmb = data.zCMB
            self._likelihood = CustomSneLikelihood(
                mag_mean, cov_mag, zhel, zcmb, no_intrinsic_scatter=True
            )
        elif sample_name == "DES5YR_Dovekie":
            from hierarc.Likelihood.SneLikelihood.des_5yr_dovekie import (
                DES5YRDovekieData,
            )

            data = DES5YRDovekieData()
            mag_mean = data.mu_obs
            cov_mag = data.cov_mag_b
            zhel = data.zHEL
            zcmb = data.zCMB
            self._likelihood = CustomSneLikelihood(
                mag_mean, cov_mag, zhel, zcmb, no_intrinsic_scatter=True
            )
        else:
            self._likelihood = SneLikelihoodFromFile(
                sample_name=sample_name, **kwargs_sne_likelihood
            )
        self.zhel = self._likelihood.zhel
        self.zcmb = self._likelihood.zcmb

    def log_likelihood(self, cosmo, apparent_m_z=None, sigma_m_z=None, z_anchor=0.1):
        """

        :param cosmo: instance of a class to compute angular diameter distances on arrays
        :param apparent_m_z: mean apparent magnitude of SN Ia at z=z_anchor (optional)
        :param z_anchor: redshift where definition of apparent_m_z is set (only applicable when apparent_m_z != None)
        :param sigma_m_z: 1-sigma scatter in magnitude in the intrinsic SNe brightness distribution not accounted-for
         by the covariance matrix
        :return: log likelihood of the data given the specified cosmology
        """
        angular_diameter_distances = cosmo.angular_diameter_distance(self.zcmb).value
        lum_dists = 5 * np.log10(
            (1 + self.zhel) * (1 + self.zcmb) * angular_diameter_distances
        )

        ang_dist_anchor = cosmo.angular_diameter_distance(z_anchor).value
        lum_dist_anchor = 5 * np.log10(
            (1 + z_anchor) * (1 + z_anchor) * ang_dist_anchor
        )

        return self._likelihood.log_likelihood_lum_dist(
            lum_dists - lum_dist_anchor, apparent_m_z, sigma_m_z
        )
