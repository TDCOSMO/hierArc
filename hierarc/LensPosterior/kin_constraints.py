import copy
import numpy as np
from hierarc.LensPosterior.base_config import BaseLensConfig
from lenstronomy.Util.param_util import ellipticity2phi_q, phi_q2_ellipticity
from itertools import product
from tqdm import tqdm, trange
import warnings


# smallest intrinsic axis ratio jampy accepts when deprojecting an MGE
# (jampy.axi.jam_axi_proj: qmin = 0.05, 'Inclination too low')
Q_INTRINSIC_MIN_JAM = 0.05


def min_q_intrinsic(q_reference, q_other, q_intrinsic_min=Q_INTRINSIC_MIN_JAM):
    """Smallest q_intrinsic for which both the light and the total mass can be deprojected.

    The inclination is a property of the galaxy, so the light and the total mass are
    deprojected at the same angle i, and each of them admits a real deprojection
    q_intr = sqrt(q_obs^2 - cos^2 i) / sin i only for cos i <= q_obs. The inclination is
    derived from one of the two (the 'reference', i.e. the total mass, see
    KinConstraints._get_inclination_angle), so when the other component is *flatter* on
    the sky it sets a lower bound on q_intrinsic: below it the model asks for an
    inclination at which the other component would have to be thinner than
    q_intrinsic_min, which is unphysical and which jampy rejects outright.

    Requiring q_other^2 >= cos^2 i + q_intrinsic_min^2 sin^2 i and substituting
    cos^2 i = (q_reference^2 - q_intrinsic^2) / (1 - q_intrinsic^2) gives the bound below.

    :param q_reference: observed axis ratio q_intrinsic is defined on (the total mass)
    :param q_other: observed axis ratio of the other component (the light)
    :param q_intrinsic_min: smallest intrinsic axis ratio the deprojection may produce
    :return: lower bound on q_intrinsic; 0 when the reference is the flatter component,
        in which case the bound is never binding
    """
    if q_other >= q_reference:
        return 0.0
    if q_other <= q_intrinsic_min:
        # the deprojection can only make a component flatter (q_intr = q_obs edge-on, less
        # otherwise), so a component observed flatter than q_intrinsic_min has no valid
        # deprojection at any inclination and no bound would rescue it
        raise ValueError(
            "the flatter component is observed at q=%s, at or below the minimum intrinsic "
            "axis ratio %s: no inclination can deproject it." % (q_other, q_intrinsic_min)
        )
    c_max = (q_other**2 - q_intrinsic_min**2) / (1 - q_intrinsic_min**2)
    return float(np.sqrt(max((q_reference**2 - c_max) / (1 - c_max), 0.0)))


class KinConstraints(BaseLensConfig):
    """Class that manages constraints from Integral Field Unit spectral observations.

    using axisymmetric Jeans modeling (JAM, Cappellari 2008) with JamPy to compute the
    dimensionless kinematic component J()
    """

    def __init__(
        self,
        z_lens,
        z_source,
        theta_E,
        theta_E_error,
        gamma,
        gamma_error,
        r_eff,
        r_eff_error,
        sigma_v_measured,
        kwargs_aperture,
        kwargs_seeing,
        anisotropy_model,
        kwargs_numerics_galkin=None,
        kwargs_numerics_jampy=None,
        axial_symmetry="spherical",
        kinematics_backend="jampy",
        q_total_mass=None,
        sigma_v_error_independent=None,
        sigma_v_error_covariant=None,
        sigma_v_error_cov_matrix=None,
        kwargs_lens_light=None,
        lens_light_model_list=None,
        lens_model_list=None,
        MGE_light=None,
        MGE_mass=None,
        kwargs_mge_light=None,
        kwargs_mge_mass=None,
        sampling_number=1000,
        num_psf_sampling=100,
        num_kin_sampling=1000,
        supersampling_factor=None,
        multi_observations=False,
        multi_light_profile=False,
        cosmo_fiducial=None,
        gamma_in_scaling=None,
        log_m2l_scaling=None,
        gamma_pl_scaling=None,
        q_intrinsic_scaling=None,
    ):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param theta_E: Einstein radius (in arc seconds)
        :param theta_E_error: 1-sigma error on Einstein radius
        :param gamma: power-law slope (2 = isothermal) estimated from imaging data
        :param gamma_error: 1-sigma uncertainty on power-law slope
        :param r_eff: half-light radius of the deflector (arc seconds)
        :param r_eff_error: uncertainty on half-light radius
        :param sigma_v_measured: numpy array of IFU velocity dispersion of the main
            deflector in km/s
        :param sigma_v_error_independent: numpy array of 1-sigma uncertainty in velocity
            dispersion of the IFU
         observation independent of each other
        :param sigma_v_error_covariant: covariant error in the measured kinematics
            shared among all IFU measurements
        :param sigma_v_error_cov_matrix: error covariance matrix in the sigma_v
            measurements (km/s)^2
        :type sigma_v_error_cov_matrix: nxn matrix with n the length of the
            sigma_v_measured array
        :param kwargs_aperture: spectroscopic aperture keyword arguments, see
            lenstronomy.Galkin.aperture for options
        :param kwargs_seeing: seeing condition of spectroscopic observation, corresponds
            to kwargs_psf in the GalKin
         module specified in lenstronomy.GalKin.psf
        :param anisotropy_model: type of stellar anisotropy model. See details in
            MamonLokasAnisotropy() class of lenstronomy.GalKin.anisotropy
        :param kwargs_numerics_galkin: numerical settings for the integrated
            line-of-sight velocity dispersion
        :param kwargs_numerics_jampy: additional kwargs for the jampy call
        :param axial_symmetry: axial symmetry assumption for JAM modeling, either 'spherical', 'axi_sph' or 'axi_cyl'.
        :param kinematics_backend: backend to compute the JAM kinematics, either 'jampy' or 'galkin'
        :param q_total_mass: float between 0 and 1, axial ratio for the total mass (stars + dark matter).
            If None, the total q is set to the same as the light profile q.
        :param lens_model_list: keyword argument list of lens model (optional)
        :param kwargs_lens_light: keyword argument list of lens light model.
            These kwargs should be provided for axisymmetric modeling to specify the light ellipticity.
        :param lens_light_model_list: list of lens light model types (optional, default is HERNQUIST)
        :param kwargs_mge_light: keyword arguments that go into the MGE decomposition
            routine
        :param multi_observations: bool, if True, interprets kwargs_aperture and
            kwargs_seeing as lists of multiple observations
        :param multi_light_profile: bool, if True (and if multi_observation=True) then treats the light profile input
         as a list for each individual observation condition.
        :param cosmo_fiducial: astropy.cosmology instance, if None,
            uses astropy's default
        :param gamma_in_scaling: array of gamma_in parameter to be interpolated (optional, otherwise None)
        :param log_m2l_scaling: array of log_m2l parameter to be interpolated (optional, otherwise None)
        :param gamma_pl_scaling: array of mass density profile power-law slope values (optional, otherwise None)
        :param q_intrinsic_scaling: array of intrinsic axis ratio values (optional, otherwise None)
            this is used for axisymmetric JAM models to get the inclination angle from the observed axis ratio
        """
        if lens_light_model_list is None:
            lens_light_model_list = ["HERNQUIST"]
        self._sigma_v_measured = np.array(sigma_v_measured)
        self._sigma_v_error_independent = np.array(sigma_v_error_independent)
        self._sigma_v_error_covariant = sigma_v_error_covariant
        self._sigma_v_error_cov_matrix = sigma_v_error_cov_matrix
        self._anisotropy_model = anisotropy_model
        self._supersampling_factor = supersampling_factor

        BaseLensConfig.__init__(
            self,
            z_lens=z_lens,
            z_source=z_source,
            theta_E=theta_E,
            theta_E_error=theta_E_error,
            gamma=gamma,
            gamma_error=gamma_error,
            r_eff=r_eff,
            r_eff_error=r_eff_error,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            anisotropy_model=anisotropy_model,
            kwargs_numerics_galkin=kwargs_numerics_galkin,
            kwargs_numerics_jampy=kwargs_numerics_jampy,
            axial_symmetry=axial_symmetry,
            kinematics_backend=kinematics_backend,
            lens_model_list=lens_model_list,
            kwargs_lens_light=kwargs_lens_light,
            lens_light_model_list=lens_light_model_list,
            MGE_light=MGE_light,
            MGE_mass=MGE_mass,
            kwargs_mge_light=kwargs_mge_light,
            kwargs_mge_mass=kwargs_mge_mass,
            sampling_number=sampling_number,
            num_psf_sampling=num_psf_sampling,
            num_kin_sampling=num_kin_sampling,
            multi_observations=multi_observations,
            multi_light_profile=multi_light_profile,
            cosmo_fiducial=cosmo_fiducial,
            gamma_in_scaling=gamma_in_scaling,
            log_m2l_scaling=log_m2l_scaling,
            gamma_pl_scaling=gamma_pl_scaling,
            q_intrinsic_scaling=q_intrinsic_scaling,
        )

        self._kwargs_mass_geometry = {}
        self._lens_phi = 0.0
        self._q_light = 1.0
        self._q_mass = 1.0

        if (self._kwargs_lens_light is not None) and (axial_symmetry != "spherical"):
            if self._multi_observations:
                kwargs_light_0 = self._kwargs_lens_light[0][0]
            else:
                kwargs_light_0 = self._kwargs_lens_light[0]
            if "e1" not in kwargs_light_0:
                raise ValueError(
                    "light ellipticities must be provided in 'kwargs_lens_light' for axisymmetric modeling."
                )
            self._lens_phi, self._q_light = ellipticity2phi_q(
                kwargs_light_0["e1"],
                kwargs_light_0["e2"],
            )
            if q_total_mass is None:
                self._q_mass = self._q_light
            else:
                self._q_mass = q_total_mass
            mass_e1, mass_e2 = phi_q2_ellipticity(self._lens_phi, self._q_mass)
            self._kwargs_mass_geometry.update(
                {
                    "e1": mass_e1,
                    "e2": mass_e2,
                }
            )
        elif (self._kwargs_lens_light is not None) and (axial_symmetry == "spherical"):
            if self._multi_observations:
                for kwargs_obs in self._kwargs_lens_light:
                    for kwargs in kwargs_obs:
                        if "e1" in kwargs:
                            kwargs.pop("e1")
                            kwargs.pop("e2")
            else:
                for kwargs in self._kwargs_lens_light:
                    if "e1" in kwargs:
                        kwargs.pop("e1")
                        kwargs.pop("e2")
        elif (self._kwargs_lens_light is None) and (axial_symmetry != "spherical"):
            raise ValueError(
                "'kwargs_lens_light' must be provided for axisymmetric modeling."
            )

    def j_kin_draw(
        self, kwargs_anisotropy, gamma_pl=None, q_intrinsic=1.0, no_error=False
    ):
        """One simple sampling realization of the dimensionless kinematics of the model.

        :param kwargs_anisotropy: keyword argument of anisotropy setting
        :param gamma_pl: power law slope, if None, draws from measurement uncertainty,
            otherwise takes at fixed value
        :param q_intrinsic: intrinsic axis ratio of the light profile to compute the
            inclination angle
        :type gamma_pl: float or None
        :param no_error: bool, if True, does not render from the uncertainty but uses
            the mean values instead
        :return: dimensionless kinematic component J() Birrer et al. 2016, 2019
        """
        theta_E_draw, gamma_draw, r_eff_draw, delta_r_eff = self.draw_lens(
            gamma_pl=gamma_pl, no_error=no_error
        )
        if self._kwargs_lens_light is None:
            kwargs_light = [{"Rs": r_eff_draw * 0.551, "amp": 1.0}]
        else:
            if self._multi_observations:
                kwargs_light = []
                for kwargs_obs in self._kwargs_lens_light:
                    kwargs_light.append(
                        self._scale_light_radius(kwargs_obs, delta_r_eff)
                    )
            else:
                kwargs_light = self._scale_light_radius(
                    self._kwargs_lens_light, delta_r_eff
                )
        kwargs_lens = [
            # add geometry for axisymmetric modeling
            {"theta_E": theta_E_draw, "gamma": gamma_draw}
            | self._kwargs_mass_geometry
        ]
        # get the inclination angle from the mass axial ratio
        self._validate_deprojection(q_intrinsic)
        inclination = self._get_inclination_angle(
            q_obs=self._q_mass, q_intrinsic=q_intrinsic
        )
        j_kin = self.velocity_dispersion_map_dimension_less(
            kwargs_lens=kwargs_lens,
            kwargs_lens_light=kwargs_light,
            kwargs_anisotropy=kwargs_anisotropy,
            inclination=inclination,
            r_eff=r_eff_draw,
            theta_E=theta_E_draw,
            gamma=gamma_draw,
            supersampling_factor=self._supersampling_factor,
        )
        return j_kin

    def hierarchy_configuration(self, num_sample_model=20):
        """Routine to configure the likelihood to be used in the hierarchical sampling.
        In particular, a default configuration is set to compute the Gaussian
        approximation of Ds/Dds by sampling the posterior and the estimate of the
        variance of the sample. The anisotropy scaling is then performed. Different
        anisotropy models are supported.

        :param num_sample_model: number of samples drawn from the lens and light model
            posterior to compute the dimensionless kinematic component J()
        :return: keyword arguments
        """

        j_model_list, error_cov_j_sqrt = self.model_marginalization(num_sample_model)
        ani_scaling_array_list = self.anisotropy_scaling()
        error_cov_measurement = self.error_cov_measurement
        # configuration keyword arguments for the hierarchical sampling
        kwargs_likelihood = {
            "z_lens": self._z_lens,
            "z_source": self._z_source,
            "likelihood_type": "IFUKinCov",
            "sigma_v_measurement": self._sigma_v_measured,
            "anisotropy_model": self._anisotropy_model,
            "j_model": j_model_list,
            "error_cov_measurement": error_cov_measurement,
            "error_cov_j_sqrt": error_cov_j_sqrt,
            "kin_scaling_param_list": self.param_name_list,
            "j_kin_scaling_param_axes": self.kin_scaling_param_array,
            "j_kin_scaling_grid_list": ani_scaling_array_list,
        }
        prior_list = []
        if "gamma_pl" in self.param_name_list:
            prior_list.append(["gamma_pl", self._gamma, self._gamma_error])
        # TODO: make sure to add other priors if needed or available
        # if "gamma_in" in self._param_name_list:
        #    prior_list.append(["gamma_in"])
        kwargs_likelihood["prior_list"] = prior_list
        # if "gamma_pl" in self.param_name_list:
        #    kwargs_likelihood["gamma_pl_sampling"] = True
        return kwargs_likelihood

    def model_marginalization(self, num_sample_model=20):
        """

        :param num_sample_model: number of samples drawn from the lens and light model
            posterior to compute the dimensionless kinematic component J()
        :return: J() as array for each measurement prediction, covariance matrix in
            sqrt(J)
        """
        num_data = len(self._sigma_v_measured)
        j_kin_matrix = np.zeros(
            (num_sample_model, num_data)
        )  # matrix that contains the sampled J() distribution
        for i in trange(num_sample_model, desc="Kinematics model marginalization"):
            j_kin = self.j_kin_draw(
                self.kwargs_anisotropy_base,
                no_error=False,
                **self.kwargs_lens_base,
                **self.kwargs_deprojection_base,
            )
            j_kin_matrix[i, :] = j_kin

        error_cov_j_sqrt = np.cov(np.sqrt(j_kin_matrix.T))
        j_model_list = np.mean(j_kin_matrix, axis=0)
        return j_model_list, error_cov_j_sqrt

    @property
    def error_cov_measurement(self):
        """Error covariance matrix of the measured velocity dispersion data points This
        is either calculated from the diagonal 'sigma_v_error_independent' and the off-
        diagonal 'sigma_v_error_covariant' terms, or directly from the
        'sigma_v_error_cov_matrix' if provided.

        :return: nxn matrix of the error covariances in the velocity dispersion
            measurements (km/s)^2
        """
        if self._sigma_v_error_cov_matrix is None:
            if (
                self._sigma_v_error_independent is None
                or self._sigma_v_error_covariant is None
            ):
                raise ValueError(
                    "sigma_v_error_independent and sigma_v_error_covariant need to be provided as arrays "
                    "of the same length as sigma_v_measurement."
                )
            error_covariance_array = (
                np.ones_like(self._sigma_v_error_independent)
                * self._sigma_v_error_covariant
            )
            error_cov_measurement = np.outer(
                error_covariance_array, error_covariance_array
            ) + np.diag(self._sigma_v_error_independent**2)
            return error_cov_measurement
        else:
            return self._sigma_v_error_cov_matrix

    def anisotropy_scaling(self):
        """

        :return: anisotropy scaling grid along the axes defined by ani_param_array
        """
        j_ani_0 = self.j_kin_draw(
            self.kwargs_anisotropy_base,
            no_error=True,
            **self.kwargs_lens_base,
            **self.kwargs_deprojection_base,
        )
        return self._anisotropy_scaling_relative(j_ani_0)

    def _anisotropy_scaling_relative(self, j_ani_0):
        """Anisotropy scaling relative to a default J prediction.

        :param j_ani_0: default J() prediction for default anisotropy
        :return: list of arrays (for the number of measurements) according to anisotropy
            scaling
        """
        num_data = len(self._sigma_v_measured)
        shapes = [len(a) for a in self.kin_scaling_param_array]
        ani_scaling_array_list = [np.zeros(shapes) for _ in range(num_data)]

        def compute_for_params(param_array, idx):
            """Computes one model for a given set of parameters in the anisotropy
            scaling grid and stores the scaling relative to the default J prediction.
            The output is stored in ani_scaling_array_list at position idx.

            :param param_array: a single set of arguments to evaluate
            :param idx: position of the model within the grid
            :return: Scaling of the dimensionless kinematic component J_i/J_0
            """
            kwargs_ani, kwargs_lens, kwargs_deprojection = self.param_array2kwargs(
                param_array=param_array
            )
            kwargs_anisotropy = self.anisotropy_kwargs(**kwargs_ani)
            j_kin_ani = self.j_kin_draw(
                kwargs_anisotropy,
                no_error=True,
                **kwargs_lens,
                **kwargs_deprojection,
            )
            for s, j_kin in enumerate(j_kin_ani):
                ani_scaling_array_list[s][idx] = j_kin / j_ani_0[s]

        if not 1 <= self.num_scaling_dim <= 4:
            raise ValueError(
                f"Kin scaling with parameter dimension {self.num_scaling_dim} "
                "not supported, choose between 1 and 4."
            )

        # total number of iterations
        total = np.prod(shapes)
        iterator = product(*[range(n) for n in shapes])
        for idx in tqdm(iterator, total=total, desc="Computing kinematics model grid"):
            param_array = [
                self.kin_scaling_param_array[d][i] for d, i in enumerate(idx)
            ]
            compute_for_params(param_array, idx if len(idx) > 1 else idx[0])

        return ani_scaling_array_list

    @property
    def q_intrinsic_min(self):
        """Lower bound of the physically valid q_intrinsic range for this lens.

        q_intrinsic is defined on the total mass, which is what the inclination is derived
        from; when the light is flatter on the sky it bounds the inclination and therefore
        q_intrinsic. Use it to build the q_intrinsic interpolation grid, so that the
        sampler never explores a range the model cannot represent.

        :return: minimum q_intrinsic, 0 when the light is the rounder component
        """
        if self.axial_symmetry == "spherical" or self._q_mass == 1.0:
            # a circular total mass carries no inclination information, and
            # _get_inclination_angle() falls back to the spherical solver, which does not
            # deproject anything: no bound applies
            return 0.0
        return min_q_intrinsic(self._q_mass, self._q_light)

    def _validate_deprojection(self, q_intrinsic):
        """Fail with a physical explanation rather than deep inside jampy.

        :param q_intrinsic: intrinsic axis ratio of the total mass
        :raises ValueError: if the light cannot be deprojected at the resulting inclination
        """
        if self.axial_symmetry == "spherical" or q_intrinsic == 1.0:
            return
        q_min = self.q_intrinsic_min
        if q_intrinsic < q_min:
            raise ValueError(
                "q_intrinsic=%.4f is below the smallest value this lens admits (%.4f). "
                "The inclination is set by the total mass (q_mass=%.4f), and at this "
                "q_intrinsic it would be low enough that the light (q_light=%.4f), which "
                "is flatter on the sky, would have to be intrinsically thinner than %.2f "
                "to project to its observed shape. Restrict the q_intrinsic interpolation "
                "grid of this lens to [%.4f, 1] (and the sampling bounds with it)."
                % (
                    q_intrinsic,
                    q_min,
                    self._q_mass,
                    self._q_light,
                    Q_INTRINSIC_MIN_JAM,
                    q_min,
                )
            )

    def _get_inclination_angle(self, q_obs, q_intrinsic):
        """Compute inclination angle from observed ellipticity and intrinsic axis ratio.

        :param q_obs: observed axis ratio
        :param q_intrinsic: intrinsic axis ratio
        :return: inclination angle in degrees
        """
        if (self.axial_symmetry == "spherical") or (q_intrinsic == 1.0):
            return 90.0  # spherical case
        if q_obs == 1.0:
            warnings.warn(
                "Cannot determine inclination angle for circular observed profile (q_obs=1.0)."
                " Spherical symmetry will be assumed.",
                UserWarning,
            )
            return None
        cos_i_squared = (q_obs**2 - q_intrinsic**2) / (1 - q_intrinsic**2)
        cos_i_squared = np.clip(cos_i_squared, 0, 1)
        inclination_angle = np.arccos(np.sqrt(cos_i_squared))
        return np.rad2deg(inclination_angle)

    @staticmethod
    def _scale_light_radius(kwargs_light, delta_r_eff):
        """Scale the radius of the light profile by a factor delta_r_eff.

        :param kwargs_light: list of keyword arguments for the light profile
        :param delta_r_eff: scaling factor for the effective radius
        :return: scaled kwargs_light
        """
        kwargs_light = copy.deepcopy(kwargs_light)
        for kwargs in kwargs_light:
            if "Rs" in kwargs:
                kwargs["Rs"] *= delta_r_eff
            if "R_sersic" in kwargs:
                kwargs["R_sersic"] *= delta_r_eff
            if "sigma" in kwargs:
                kwargs["sigma"] *= delta_r_eff
        return kwargs_light
