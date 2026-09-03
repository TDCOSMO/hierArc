import numpy as np
import pytest

from hierarc.Likelihood.hierarchy_likelihood import LensLikelihood
from hierarc.Likelihood.quadrature_marginalisation import (
    QuadratureMarginalisation,
    QuadratureNotApplicable,
)
from hierarc.Util.quadrature_util import (
    truncated_normal_panels,
    histogram_nodes,
    gauss_legendre_panels,
)


def _kwargs_lens(num_draws=200, **kwargs):
    """A DdtHistKin lens with a 3d kinematics scaling grid and a kappa histogram."""
    np.random.seed(42)
    num_bins = 6
    axes = [
        np.linspace(-0.49, 1.0, 7),  # a_ani (interpolated as beta)
        np.linspace(1.5, 2.5, 5),  # gamma_pl
        np.linspace(0.2, 1.0, 5),  # q_intrinsic
    ]
    shape = tuple(len(a) for a in axes)
    grid_list = [
        1.0 + 0.4 * np.arange(1, num_bins + 1)[i % num_bins] / num_bins
        + 0.3 * np.add.outer(np.add.outer(axes[0], axes[1]), axes[2])
        for i in range(num_bins)
    ]
    j_model = np.linspace(0.9, 1.1, num_bins)
    sigma_v = 250 + np.arange(num_bins) * 2.0
    cov_meas = np.diag((sigma_v * 0.05) ** 2) + 25.0
    cov_j = np.diag(j_model * 1e-3) + 1e-4
    kappa_edges = np.linspace(-0.05, 0.15, 41)
    kappa_mid = 0.5 * (kappa_edges[:-1] + kappa_edges[1:])
    kappa_pdf = np.exp(-0.5 * ((kappa_mid - 0.02) / 0.03) ** 2)

    base = dict(
        z_lens=0.3,
        z_source=2.0,
        name="TEST",
        likelihood_type="DdtHistKin",
        ddt_samples=np.random.normal(2500, 150, 4000),
        sigma_v_measurement=sigma_v,
        j_model=j_model,
        error_cov_measurement=cov_meas,
        error_cov_j_sqrt=cov_j,
        kin_scaling_param_list=["a_ani", "gamma_pl", "q_intrinsic"],
        j_kin_scaling_param_axes=axes,
        j_kin_scaling_grid_list=grid_list,
        anisotropy_model="const",
        anisotropy_sampling=True,
        anisotropy_distribution="GAUSSIAN",
        anisotropy_parameterization="TAN_RAD",
        q_intrinsic_sampling=True,
        q_intrinsic_distribution="GAUSSIAN",
        lambda_mst_distribution="GAUSSIAN",
        los_distribution_individual="PDF",
        kwargs_los_individual={"bin_edges": kappa_edges, "pdf_array": kappa_pdf},
        gamma_pl_index=0,
        num_distribution_draws=num_draws,
        normalized=False,
    )
    base.update(kwargs)
    return base


KWARGS_LENS = dict(lambda_mst=1.0, lambda_mst_sigma=0.05, gamma_pl_list=[2.0])
KWARGS_KIN = dict(a_ani=1.0, a_ani_sigma=0.1, q_intrinsic=0.7, q_intrinsic_sigma=0.08)
DDT, DD = 3000.0, 1200.0


class TestQuadratureUtil(object):
    def test_truncated_normal_panels(self):
        nodes, weights = truncated_normal_panels([-6, -1, 0, 2, 6], 0.2, 0.7, 8)
        npt = np.testing
        npt.assert_almost_equal(np.sum(weights), 1.0, decimal=6)
        npt.assert_almost_equal(np.sum(weights * nodes), 0.2, decimal=5)
        npt.assert_almost_equal(
            np.sum(weights * (nodes - 0.2) ** 2), 0.7**2, decimal=5
        )

    def test_truncated_normal_panels_drops_empty(self):
        # a panel 40 sigma away holds no mass and should not contribute nodes
        many, _ = truncated_normal_panels([-50, -40, -3, 3], 0.0, 1.0, 4, mass_tol=0)
        few, _ = truncated_normal_panels([-50, -40, -3, 3], 0.0, 1.0, 4, mass_tol=1e-8)
        assert len(few) < len(many)

    def test_truncated_normal_panels_zero_sigma(self):
        nodes, weights = truncated_normal_panels([-1, 1], 0.3, 0.0, 4)
        assert nodes == pytest.approx([0.3])
        assert weights == pytest.approx([1.0])

    def test_histogram_nodes(self):
        edges = np.linspace(0, 2, 21)
        nodes, weights = histogram_nodes(edges, np.ones(20), n_per_bin=3)
        npt = np.testing
        npt.assert_almost_equal(np.sum(weights), 1.0, decimal=12)
        npt.assert_almost_equal(np.sum(weights * nodes), 1.0, decimal=10)

    def test_gauss_legendre_panels(self):
        nodes, weights = gauss_legendre_panels([0, 1, 3], 5)
        np.testing.assert_almost_equal(np.sum(weights * nodes**4), 3**5 / 5, decimal=8)


class TestPencil(object):
    def test_matches_direct_evaluation(self):
        lens = LensLikelihood(**_kwargs_lens())
        kin = lens._lens_type._kinlikelihood
        scaling = np.linspace(0.9, 1.2, kin.num_data)
        pencil = kin.pencil_setup(kin_scaling=scaling)
        ds_dds = np.linspace(0.5, 2.5, 17)
        fast = kin.log_likelihood_pencil(ds_dds, pencil)
        slow = [
            kin.log_likelihood(t * DD * (1 + kin._z_lens), DD, kin_scaling=scaling)
            for t in ds_dds
        ]
        np.testing.assert_allclose(fast, slow, atol=1e-8)

    def test_normalized(self):
        kwargs = _kwargs_lens()
        kwargs["normalized"] = True
        lens = LensLikelihood(**kwargs)
        kin = lens._lens_type._kinlikelihood
        scaling = np.ones(kin.num_data) * 1.05
        pencil = kin.pencil_setup(kin_scaling=scaling)
        ds_dds = np.array([0.8, 1.4])
        fast = kin.log_likelihood_pencil(ds_dds, pencil)
        slow = [
            kin.log_likelihood(t * DD * (1 + kin._z_lens), DD, kin_scaling=scaling)
            for t in ds_dds
        ]
        np.testing.assert_allclose(fast, slow, atol=1e-7)


class TestQuadratureMarginalisation(object):
    def test_default_is_monte_carlo(self):
        lens = LensLikelihood(**_kwargs_lens())
        assert lens._marginalisation == "mc"
        assert lens._quadrature is None

    def test_agrees_with_monte_carlo(self):
        """The two marginalisations must agree within the Monte Carlo error."""
        mc = LensLikelihood(**_kwargs_lens(num_draws=20000))
        quad = LensLikelihood(**_kwargs_lens(marginalisation="quadrature"))
        assert quad._quadrature is not None
        kwargs = dict(
            kwargs_lens=KWARGS_LENS, kwargs_kin=KWARGS_KIN, kwargs_source={},
            kwargs_los=[],
        )
        np.random.seed(11)
        value_mc = mc.hyper_param_likelihood(DDT, DD, 0, beta_dsp=None, **kwargs)
        value_quad = quad.hyper_param_likelihood(DDT, DD, 0, beta_dsp=None, **kwargs)
        # 20000 draws leaves a small residual scatter; 0.05 nats is a loose but
        # meaningful bound and would catch any structural error
        assert abs(value_mc - value_quad) < 0.05

    def test_converges(self):
        """Refining the rule must converge, and the error must fall."""
        reference = None
        errors = []
        for n_gauss in (2, 3, 5, 8):
            lens = LensLikelihood(
                **_kwargs_lens(
                    marginalisation="quadrature",
                    kwargs_marginalisation={"n_gauss": n_gauss, "n_lambda_tot": 512},
                )
            )
            value = lens.hyper_param_likelihood(
                DDT, DD, 0, beta_dsp=None, kwargs_lens=KWARGS_LENS,
                kwargs_kin=KWARGS_KIN, kwargs_source={}, kwargs_los=[],
            )
            if n_gauss == 8:
                reference = value
            else:
                errors.append(value)
        errors = [abs(v - reference) for v in errors]
        assert errors[0] > errors[-1]
        assert errors[-1] < 1e-3

    def test_deterministic(self):
        """The whole point: the same inputs must give exactly the same number."""
        lens = LensLikelihood(**_kwargs_lens(marginalisation="quadrature"))
        values = []
        for seed in (1, 2, 3):
            np.random.seed(seed)
            values.append(
                lens.hyper_param_likelihood(
                    DDT, DD, 0, beta_dsp=None, kwargs_lens=KWARGS_LENS,
                    kwargs_kin=KWARGS_KIN, kwargs_source={}, kwargs_los=[],
                )
            )
        assert values[0] == values[1] == values[2]

    def test_rejects_unsupported(self):
        with pytest.raises(QuadratureNotApplicable):
            LensLikelihood(
                **_kwargs_lens(
                    marginalisation="quadrature",
                    anisotropy_model="OM",
                    kin_scaling_param_list=["a_ani", "gamma_pl", "q_intrinsic"],
                )
            )

    def test_fallback(self):
        """'quadrature_if_applicable' warns and falls back rather than failing."""
        with pytest.warns(UserWarning):
            lens = LensLikelihood(
                **_kwargs_lens(
                    marginalisation="quadrature_if_applicable", anisotropy_model="OM"
                )
            )
        assert lens._quadrature is None
        value = lens.hyper_param_likelihood(
            DDT, DD, 0, beta_dsp=None, kwargs_lens=KWARGS_LENS,
            kwargs_kin=KWARGS_KIN, kwargs_source={}, kwargs_los=[],
        )
        assert np.isfinite(value)

    def test_bad_marginalisation_name(self):
        with pytest.raises(ValueError):
            LensLikelihood(**_kwargs_lens(marginalisation="nonsense"))

    def test_ifu_kin_cov(self):
        """A kinematics-only lens has no Ddt term but the same covariance structure."""
        kwargs = _kwargs_lens(num_draws=20000)
        kwargs.pop("ddt_samples")
        kwargs["likelihood_type"] = "IFUKinCov"
        mc = LensLikelihood(**kwargs)
        kwargs_q = dict(kwargs)
        kwargs_q["marginalisation"] = "quadrature"
        quad = LensLikelihood(**kwargs_q)
        assert quad._quadrature is not None
        args = dict(
            kwargs_lens=KWARGS_LENS, kwargs_kin=KWARGS_KIN, kwargs_source={},
            kwargs_los=[],
        )
        np.random.seed(5)
        value_mc = mc.hyper_param_likelihood(DDT, DD, 0, beta_dsp=None, **args)
        value_quad = quad.hyper_param_likelihood(DDT, DD, 0, beta_dsp=None, **args)
        assert abs(value_mc - value_quad) < 0.05

    def test_num_nodes(self):
        lens = LensLikelihood(
            **_kwargs_lens(
                marginalisation="quadrature", kwargs_marginalisation={"n_gauss": 3}
            )
        )
        lens.hyper_param_likelihood(
            DDT, DD, 0, beta_dsp=None, kwargs_lens=KWARGS_LENS,
            kwargs_kin=KWARGS_KIN, kwargs_source={}, kwargs_los=[],
        )
        assert lens._quadrature.num_nodes > 0
