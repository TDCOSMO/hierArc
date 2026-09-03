"""Deterministic marginalisation over the per-lens latent parameters.

hierArc marginalises each lens over the latent parameters its population distributions
describe by drawing ``num_distribution_draws`` samples and averaging. That estimator is
unbiased in the likelihood, so the sampler still targets the right posterior, but its
variance can be severe: when a lens's own data constrain the latent parameters far more
tightly than the population does, almost all the weight lands on a single draw and the
log likelihood acquires several nats of noise. This module replaces the average by a
quadrature rule, which is deterministic and converges geometrically.

The scheme has three parts.

1. **Reduction.** ``lambda_mst`` and ``kappa_ext`` reach the data only through their
   product ``lambda_tot``, so the pair collapses to one variable whose rule is built
   once per evaluation, independent of the kinematics parameters.
2. **Factorisation.** The kinematics covariance is an affine family in Ds/Dds, so one
   eigendecomposition per (anisotropy, deprojection) node makes the whole ``lambda_tot``
   sweep cost O(n) per node instead of an O(n^3) Cholesky each; see
   :func:`~hierarc.Likelihood.LensLikelihood.kin_likelihood.KinLikelihood.pencil_setup`.
3. **Panel alignment.** The kinematics scaling J is read off its grid by multilinear
   interpolation, so the integrand is only C0, with a discontinuous derivative on the
   cell walls. Gauss rules assume smoothness and stall when laid across a kink, so the
   panels are split at the grid nodes, where the kinks are known to be.

Only the configurations listed in :func:`QuadratureMarginalisation.check_applicable` are
supported. Anything else raises :class:`QuadratureNotApplicable` rather than silently
returning a number computed under assumptions that do not hold.
"""

__author__ = "mmillon"

import numpy as np
from scipy.special import logsumexp

from hierarc.Util.quadrature_util import (
    truncated_normal_panels,
    histogram_nodes,
    gauss_legendre_panels,
)

_SUPPORTED_LIKELIHOOD_TYPES = ["DdtHistKin", "IFUKinCov"]


class QuadratureNotApplicable(Exception):
    """Raised when a lens configuration is outside what the quadrature scheme covers."""


class QuadratureMarginalisation(object):
    """Deterministic replacement for the Monte Carlo loop of a single lens."""

    def __init__(
        self,
        lens,
        n_gauss=4,
        n_lambda_tot=256,
        kappa_sub_bin=2,
        n_sub_panel=None,
        panel_mass_tol=1e-8,
        lambda_sigma_range=7.0,
    ):
        """

        :param lens: the ``LensLikelihood`` this rule marginalises
        :param n_gauss: accuracy dial for the anisotropy and deprojection directions.
            Each J-grid cell is split into n_gauss sub-panels carrying n_gauss
            Gauss-Legendre nodes each, so the node count grows as n_gauss^4 and both the
            panel width and the order improve together. 2 to 4 covers the useful range.
        :param n_lambda_tot: number of nodes for the lambda_tot sweep. These are
            cheap (O(n) each); a few hundred resolve the Ddt likelihood.
        :param kappa_sub_bin: Gauss-Legendre nodes inside each bin of the external
            convergence histogram
        :param n_sub_panel: override the sub-panel count, which otherwise follows
            n_gauss. Raising the order inside a panel converges only once that panel
            resolves the integrand; for the most sharply constrained lenses the peak is
            narrower than one J-grid cell, and subdividing is what converges.
        :param panel_mass_tol: drop panels holding less than this share of the mass
        :param lambda_sigma_range: half-width, in sigma, of the lambda_mst rule
        :raises QuadratureNotApplicable: if the lens configuration is not supported
        """
        self.check_applicable(lens)
        self._lens = lens
        self._n_gauss = int(n_gauss)
        self._n_lambda_tot = int(n_lambda_tot)
        self._kappa_sub_bin = int(kappa_sub_bin)
        # h-refinement matters as much as p-refinement here: raising the order inside a
        # panel only converges once that panel resolves the integrand, and for the most
        # sharply constrained lenses the peak is narrower than one J-grid cell. Coupling
        # the two keeps n_gauss a single dial that converges every lens.
        self._n_sub_panel = int(n_gauss if n_sub_panel is None else n_sub_panel)
        self._panel_mass_tol = float(panel_mass_tol)
        self._lambda_sigma_range = float(lambda_sigma_range)

        self._kin_likelihood = _kin_likelihood_of(lens)
        self._td_likelihood = getattr(lens._lens_type, "_tdLikelihood", None)
        axes = lens.kin_scaling_axes
        names = lens.kin_scaling_param_names
        self._beta_grid = _axis_for(names, axes, "a_ani")
        self._q_grid = _axis_for(names, axes, "q_intrinsic")

    # ------------------------------------------------------------------ applicability
    @staticmethod
    def check_applicable(lens):
        """Verify that the assumptions behind the scheme hold for this lens.

        :param lens: a ``LensLikelihood``
        :raises QuadratureNotApplicable: with the reason, if any assumption fails
        """

        def reject(reason):
            raise QuadratureNotApplicable(
                "lens %s: %s. Use the Monte Carlo marginalisation for this lens."
                % (getattr(lens, "name", "?"), reason)
            )

        if lens.likelihood_type not in _SUPPORTED_LIKELIHOOD_TYPES:
            reject(
                "likelihood type '%s' is not supported (supported: %s)"
                % (lens.likelihood_type, ", ".join(_SUPPORTED_LIKELIHOOD_TYPES))
            )
        names = lens.kin_scaling_param_names
        unsupported = set(names) - {"a_ani", "gamma_pl", "q_intrinsic"}
        if unsupported:
            reject(
                "the kinematics scaling interpolates over %s, which the rule does not "
                "cover" % sorted(unsupported)
            )
        lens_dist = lens._lens_distribution
        if lens_dist._gamma_in_sampling or lens_dist._log_m2l_sampling:
            reject("gamma_in / log_m2l sampling adds latent dimensions")
        if lens_dist._gamma_pl_global_sampling and (
            lens_dist._gamma_pl_global_dist not in ["NONE"]
        ):
            reject("a global gamma_pl distribution adds a latent dimension")
        if lens_dist._lambda_mst_distribution not in ["NONE", "GAUSSIAN"]:
            reject(
                "lambda_mst distribution '%s' is not supported"
                % lens_dist._lambda_mst_distribution
            )
        aniso = lens._aniso_distribution
        if aniso._anisotropy_sampling and aniso._anisotropy_model not in ["const"]:
            reject(
                "anisotropy model '%s' is not supported (only 'const' has a scalar "
                "latent parameter)" % aniso._anisotropy_model
            )
        if aniso._anisotropy_sampling and aniso._distribution_function not in [
            "NONE",
            "GAUSSIAN",
            "GAUSSIAN_SCALED",
        ]:
            reject(
                "anisotropy distribution '%s' is not supported"
                % aniso._distribution_function
            )
        deproj = lens._deprojection_distribution
        if deproj._deprojection_sampling and deproj._distribution_function not in [
            "NONE",
            "GAUSSIAN",
            "GAUSSIAN_SCALED",
        ]:
            reject(
                "q_intrinsic distribution '%s' is not supported"
                % deproj._distribution_function
            )
        if lens._axisymmetric_correction_sampling:
            reject("the axisymmetric correction adds a latent dimension")
        los = lens._los
        if los._draw_kappa_global:
            reject("a global line-of-sight distribution is not supported")
        if los._draw_kappa_individual and not hasattr(los._kappa_dist, "bin_edges"):
            reject(
                "the individual line-of-sight distribution is not a histogram ('PDF')"
            )
        if _kin_likelihood_of(lens) is None:
            reject("no kinematics likelihood found to factorise")

    # ------------------------------------------------------------------- the 1d rules
    def _lambda_tot_rule(self, kwargs_lens, kwargs_los):
        """Nodes and weights for lambda_tot = lambda_mst * (1 - kappa_ext).

        The two enter the model only through this product, so they are collapsed into a
        single variable here: the mixture over the kappa histogram of the (scaled)
        lambda_mst normals is evaluated on one composite Gauss grid.
        """
        draw = self._lens._lens_distribution
        mean = kwargs_lens.get("lambda_mst", 1.0)
        if draw._mst_ifu:
            mean = kwargs_lens.get("lambda_ifu", 1.0)
            sigma = kwargs_lens.get("lambda_ifu_sigma", 0.0)
        else:
            sigma = kwargs_lens.get("lambda_mst_sigma", 0.0)
        mean = (
            mean
            + kwargs_lens.get("alpha_lambda", 0.0) * draw._lambda_scaling_property
            + kwargs_lens.get("beta_lambda", 0.0) * draw._lambda_scaling_property_beta
        )
        if not draw._lambda_mst_sampling:
            sigma = 0.0

        los = self._lens._los
        if los._draw_kappa_individual:
            kappa, w_kappa = histogram_nodes(
                los._kappa_dist.bin_edges,
                los._kappa_dist.pdf_array,
                n_per_bin=self._kappa_sub_bin,
            )
        else:
            kappa, w_kappa = np.array([0.0]), np.array([1.0])

        scale = 1.0 - kappa
        mu, sd = mean * scale, sigma * scale
        if sigma <= 0:
            values = np.maximum(mu, 1e-4)
            return values, w_kappa
        lo = max(np.min(mu - self._lambda_sigma_range * sd), 1e-4)
        hi = np.max(mu + self._lambda_sigma_range * sd)
        n_panel = max(int(self._n_lambda_tot) // 4, 4)
        nodes, w_plain = gauss_legendre_panels(np.linspace(lo, hi, n_panel + 1), 4)
        # density of the product, as a mixture over the kappa bins
        density = np.sum(
            w_kappa[None, :]
            * np.exp(-0.5 * ((nodes[:, None] - mu[None, :]) / sd[None, :]) ** 2)
            / (sd[None, :] * np.sqrt(2 * np.pi)),
            axis=1,
        )
        return nodes, w_plain * density

    def _outer_nodes(self, kwargs_kin):
        """Grid-aligned nodes for the anisotropy and deprojection parameters.

        The panels are split where the interpolation of J loses smoothness, i.e. at
        its grid nodes. In the anisotropy direction the sampled parameter may be the
        tangential-to-radial ratio a, related to the interpolated beta by
        beta = 1 - a^2, so one beta node becomes two edges at a = +/- sqrt(1 - beta).
        """
        aniso = self._lens._aniso_distribution
        deproj = self._lens._deprojection_distribution

        # --- anisotropy
        if not aniso._anisotropy_sampling or aniso._distribution_function == "NONE":
            beta_nodes = np.array([_beta_of(aniso, kwargs_kin.get("a_ani", None))])
            beta_weights = np.array([1.0])
        else:
            a_mean = kwargs_kin["a_ani"]
            a_sigma = kwargs_kin.get("a_ani_sigma", 0.0)
            if aniso._distribution_function == "GAUSSIAN_SCALED":
                a_sigma = a_sigma * a_mean
            if aniso._parametrization == "TAN_RAD":
                # beta = 1 - a^2, so restricting beta to [beta_min, beta_max] allows
                # sqrt(1 - beta_max) <= |a| <= sqrt(1 - beta_min): one band if
                # beta_max >= 1, otherwise two, with a hole around a = 0. Getting this
                # wrong puts nodes outside the interpolation grid.
                a_outer = np.sqrt(max(1.0 - aniso._a_ani_min, 0.0))
                a_inner = np.sqrt(max(1.0 - aniso._a_ani_max, 0.0))
                if a_inner > 0:
                    intervals = [(-a_outer, -a_inner), (a_inner, a_outer)]
                else:
                    intervals = [(-a_outer, a_outer)]
                kinks = []
                for grid_value in self._beta_grid:
                    radicand = 1.0 - grid_value
                    if radicand >= 0:
                        root = np.sqrt(radicand)
                        kinks.extend([root, -root])
                a_nodes, beta_weights = truncated_normal_panels(
                    intervals,
                    a_mean,
                    a_sigma,
                    self._n_gauss,
                    interior_edges=kinks,
                    mass_tol=self._panel_mass_tol,
                    n_sub=self._n_sub_panel,
                )
                beta_nodes = np.clip(
                    1.0 - a_nodes**2, aniso._a_ani_min, aniso._a_ani_max
                )
            else:
                beta_nodes, beta_weights = truncated_normal_panels(
                    [(aniso._a_ani_min, aniso._a_ani_max)],
                    a_mean,
                    a_sigma,
                    self._n_gauss,
                    interior_edges=self._beta_grid,
                    mass_tol=self._panel_mass_tol,
                    n_sub=self._n_sub_panel,
                )

        # --- deprojection
        if not deproj._deprojection_sampling or deproj._distribution_function == "NONE":
            q_nodes = np.array([kwargs_kin.get("q_intrinsic", None)])
            q_weights = np.array([1.0])
        else:
            q_mean = kwargs_kin["q_intrinsic"]
            q_sigma = kwargs_kin.get("q_intrinsic_sigma", 0.0)
            if deproj._distribution_function == "GAUSSIAN_SCALED":
                q_sigma = q_sigma * q_mean
            q_nodes, q_weights = truncated_normal_panels(
                [(deproj._q_min, deproj._q_max)],
                q_mean,
                q_sigma,
                self._n_gauss,
                interior_edges=self._q_grid,
                mass_tol=self._panel_mass_tol,
                n_sub=self._n_sub_panel,
            )
        return beta_nodes, beta_weights, q_nodes, q_weights

    # -------------------------------------------------------------------- the integral
    def log_likelihood(
        self,
        ddt,
        dd,
        kwargs_lens,
        kwargs_kin,
        kwargs_los=None,
        sigma_v_sys_error=None,
    ):
        """Marginal log likelihood of this lens, by quadrature.

        :param ddt: time-delay distance from the cosmology
        :param dd: angular diameter distance to the deflector
        :param kwargs_lens: lens model hyperparameters
        :param kwargs_kin: kinematics hyperparameters
        :param kwargs_los: line-of-sight hyperparameters (unused; individual only)
        :param sigma_v_sys_error: optional systematic velocity dispersion error
        :return: log likelihood, marginalised over the latent parameters
        """
        lens = self._lens
        lambda_tot, weights = self._lambda_tot_rule(kwargs_lens, kwargs_los)
        gamma_pl = lens._lens_distribution.draw_lens(**kwargs_lens).get("gamma_pl", 2)

        ddt_shifted = lens.displace_prediction(
            ddt, dd, gamma_ppn=kwargs_lens.get("gamma_ppn", 1), lambda_mst=lambda_tot,
            kappa_ext=0, mag_source=0, gamma_pl=gamma_pl,
        )[0]
        ds_dds = np.maximum(ddt_shifted / dd / (1 + lens.z_lens), 0.0)

        log_weights = np.log(np.maximum(weights, 1e-300))
        if self._td_likelihood is not None:
            log_weights = log_weights + self._td_likelihood.log_likelihood_vector(
                ddt_shifted
            )

        beta_nodes, beta_weights, q_nodes, q_weights = self._outer_nodes(kwargs_kin)
        self._num_nodes = len(beta_nodes) * len(q_nodes)
        terms = []
        for i, beta in enumerate(beta_nodes):
            for j, q_value in enumerate(q_nodes):
                kwargs_param = {
                    "lambda_mst": 1.0,
                    "gamma_ppn": 1.0,
                    "gamma_pl": gamma_pl,
                }
                if beta is not None:
                    kwargs_param["a_ani"] = beta
                if q_value is not None:
                    kwargs_param["q_intrinsic"] = q_value
                scaling = lens.kin_scaling(kwargs_param)
                pencil = self._kin_likelihood.pencil_setup(
                    kin_scaling=scaling, sigma_v_sys_error=sigma_v_sys_error
                )
                log_like = self._kin_likelihood.log_likelihood_pencil(ds_dds, pencil)
                log_prior = lens._prior.log_likelihood(kwargs_param)
                terms.append(
                    logsumexp(log_like + log_weights)
                    + np.log(beta_weights[i] * q_weights[j])
                    + log_prior
                )
        return float(logsumexp(terms))

    @property
    def num_nodes(self):
        """Number of eigendecompositions per evaluation for the current settings.

        Only known after a call, since panels below the mass tolerance are dropped.

        :return: int or None
        """
        return getattr(self, "_num_nodes", None)


def _kin_likelihood_of(lens):
    """The KinLikelihood inside a composite lens likelihood, if there is one."""
    lens_type = getattr(lens, "_lens_type", None)
    for attribute in ("_kinlikelihood", "_kin_likelihood"):
        found = getattr(lens_type, attribute, None)
        if found is not None:
            return found
    if lens_type is not None and hasattr(lens_type, "pencil_setup"):
        return lens_type
    return None


def _axis_for(names, axes, target):
    """Interpolation axis of a named kinematics-scaling parameter, or an empty one."""
    if names is None or axes is None:
        return np.array([])
    for name, axis in zip(names, axes):
        if name == target:
            return np.asarray(axis, dtype=float)
    return np.array([])


def _beta_of(aniso, a_ani):
    """The interpolated anisotropy value for a sampled parameter that is not drawn."""
    if a_ani is None:
        return None
    if aniso._parametrization == "TAN_RAD":
        return 1.0 - a_ani**2
    return a_ani
