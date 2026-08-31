import numpy as np
import numpy.testing as npt

from hierarc.Likelihood.SneLikelihood.des_5yr_dovekie import DES5YRDovekieData
from hierarc.Likelihood.SneLikelihood.sne_likelihood import SneLikelihood


class TestDES5YRDovekieData(object):
    def setup_method(self):
        pass

    def test_import(self):
        data = DES5YRDovekieData()
        mag_mean = data.mu_obs
        cov_mag = data.cov_mag_b
        zhel = data.zHEL
        zcmb = data.zCMB
        assert len(mag_mean) == 1820
        npt.assert_almost_equal(zcmb[0], 0.02509)
        npt.assert_almost_equal(zhel[0], 0.02507)
        npt.assert_almost_equal(mag_mean[0], 35.25995)
        assert len(mag_mean) == len(zhel)
        assert len(zhel) == len(zcmb)
        nx, ny = np.shape(cov_mag)
        assert nx == ny
        assert nx == len(mag_mean)
        # the released matrix is the inverse of stat+sys: it must be symmetric and
        # positive definite once inverted back
        npt.assert_allclose(cov_mag, cov_mag.T, rtol=1e-8, atol=1e-10)
        assert np.all(np.linalg.eigvalsh(cov_mag) > 0)

    def test_likelihood(self):
        from astropy.cosmology import FlatLambdaCDM

        likelihood = SneLikelihood(sample_name="DES5YR_Dovekie")
        assert len(likelihood.zcmb) == 1820

        # the absolute magnitude (and hence H0) is marginalized over
        cosmo1 = FlatLambdaCDM(H0=70, Om0=0.3)
        cosmo2 = FlatLambdaCDM(H0=67, Om0=0.3)
        logl1 = likelihood.log_likelihood(cosmo1)
        logl2 = likelihood.log_likelihood(cosmo2)
        npt.assert_almost_equal(logl1, logl2, decimal=6)

        # Om = 0.325 is preferred over clearly discrepant values
        cosmo_best = FlatLambdaCDM(H0=70, Om0=0.325)
        assert likelihood.log_likelihood(cosmo_best) > logl1
        cosmo_bad = FlatLambdaCDM(H0=70, Om0=0.2)
        assert likelihood.log_likelihood(cosmo_bad) < logl1
