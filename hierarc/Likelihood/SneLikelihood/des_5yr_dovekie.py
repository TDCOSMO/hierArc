import os
import pandas as pd
import numpy as np
import hierarc

_PATH_2_DATA = os.path.join(os.path.dirname(hierarc.__file__), "Data", "SNe")


class DES5YRDovekieData(object):
    """This class is a lightweight version of the DES-SN5YR 'Dovekie' re-calibration of
    the DES Year 5 supernova sample.

    The data and covariances that are stored in hierArc are originally from `DES 5YR Data products`_
    (``4_DISTANCES_COVMAT/DES-Dovekie_HD.csv`` and ``4_DISTANCES_COVMAT/STAT+SYS.npz``) and are read in
    exactly the way prescribed by the released `DES Dovekie likelihood`_.

    Two things differ from the original DES-SN5YR data release (see :class:`~hierarc.Likelihood.SneLikelihood.des_5yr.DES5YRData`):

    #. the covariance file stores the *inverse* stat+sys covariance matrix, in upper-triangular
       packed form, in a ``.npz`` archive;
    #. because the stored matrix is the inverse of stat+sys, the statistical uncertainties are
       already included and must **not** be added to the diagonal a second time.

    If you make use of these products, please cite `DES collaboration et al. (2024)`_ and the Dovekie
    calibration paper (Popovic et al. 2024).

    .. _DES 5YR Data products: https://github.com/des-science/DES-SN5YR/tree/main/4_DISTANCES_COVMAT
    .. _DES Dovekie likelihood: https://github.com/des-science/DES-SN5YR/blob/main/4_DISTANCES_COVMAT/DES-Dovekie-SN_Likelihood.py
    .. _DES collaboration et al. (2024): https://ui.adsabs.harvard.edu/abs/2024ApJ...973L..14D/abstract

    The Dark Energy Survey: Cosmology Results With ~1500 New High-redshift Type Ia Supernovae Using The Full 5-year Dataset. The Astrophysical Journal Letters, Volume 973, Issue 1, id.L14, 20 pp. DES Collaboration (2024).
    The Dark Energy Survey Supernova Program: Cosmological Analysis and Systematic Uncertainties. The Astrophysical Journal, Volume 975, Issue 1, id.86, 31 pp. Vincenzi et al (2024).
    Light curve and ancillary data release for the full Dark Energy Survey Supernova Program. The Astrophysical Journal, Volume 975, Issue 1, id.5, 12 pp Sánchez et al. (2024)
    """

    def __init__(self):
        self._data_file = os.path.join(
            _PATH_2_DATA, "DES-SN5YR-Dovekie", "DES-Dovekie_HD.csv"
        )
        self._cov_file = os.path.join(_PATH_2_DATA, "DES-SN5YR-Dovekie", "STAT+SYS.npz")

        print("Loading DES Y5 Dovekie SN data from {}".format(self._data_file))
        # despite the .csv extension, the Hubble diagram is released in the SNANA
        # white-space separated format with a 'VARNAMES:' header and 'SN:' row keys
        data = pd.read_csv(self._data_file, sep=r"\s+", comment="#")
        self.origlen = len(data)

        # The only columns that we actually need here are the redshift,
        # distance modulus and distance modulus error
        self.ww = data["zHD"] > 0.00
        # use the vpec corrected redshift for zCMB
        self.zCMB = data["zHD"][self.ww].to_numpy()
        self.zHEL = data["zHEL"][self.ww].to_numpy()
        # distance modulus and relative stat uncertainties.
        # ATTENTION: 'MUERR' here, not 'MUERR_FINAL' as in the original DES-SN5YR release
        self.mu_obs = data["MU"][self.ww].to_numpy()
        self.mu_obs_err = data["MUERR"][self.ww].to_numpy()

        print(
            f"Found {len(self.zCMB)} DES Y5 Dovekie supernovae (or bins if you used the binned data file)"
        )

        self.cov_mag_b = self.build_covariance()

    def build_covariance(self):
        """Run once at the start to build the covariance matrix for the data.

        The released file stores the *inverse* of the stat+sys covariance matrix in
        upper-triangular packed form (the format anticipated for the much larger LSST
        files). We unpack it, symmetrize it and invert it back to a covariance matrix,
        as done in the released DES Dovekie likelihood. The statistical uncertainties are
        already contained in it and are therefore not added to the diagonal.

        :return: covariance matrix of the distance moduli (2d numpy array)
        """
        filename = self._cov_file
        print("Loading DES Y5 Dovekie SN covariance from {}".format(filename))

        d = np.load(filename)
        n = int(d[d.files[0]][0])
        inv_cov = np.zeros((n, n))
        inv_cov[np.triu_indices(n)] = d[d.files[1]]

        # reflect the upper triangular part onto the lower one to make it symmetric
        i_lower = np.tril_indices(n, -1)
        inv_cov[i_lower] = inv_cov.T[i_lower]

        # hierArc's likelihood expects the covariance matrix, not its inverse
        cov = np.linalg.inv(inv_cov)

        cov = cov[self.ww][:, self.ww]

        return cov
