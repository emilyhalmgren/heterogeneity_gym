import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation
from cryojax.io import get_atom_info_from_mdtraj
from heterogeneity_gym import rendering
from heterogeneity_gym.hsp90.pdbs import _load_hsp90_traj
from heterogeneity_gym.likelihoods import _calculate_log_likelihood_of_atom_structures


# TODO: Reparameterize to use SNR instead of noise_std
# TODO: Add an assert that checks if atoms have escaped the imaging region.
# TODO: Type hints and documentation
# TODO: Initialize reflection tensor in __init__


class DiscreteClassModel:
    def __init__(
        self,
        atom_positions,
        identities,
        b_factors=None,
        latent_density=None,
        image_width_in_pixels: int = 128,
        pixel_size: float = 1.1,
        defocus_range: Tuple[float, float] = (5000.0, 10000.0),
        astigmatism_range: Tuple[float, float] = (0.0, 0.0),
        voltage_in_kilovolts: float = 300.0,
        noise_std: float = 0.0,
        seed: int = 0,
    ):
        """
        TODO: we should construct a "default" latent density.
        TODO: Add initialization to device.  How does OpenAI solve device when creating their "environments"?
        """
        if b_factors is None:
            b_factors = jnp.ones_like(identities) * 5.0  # Arbitrarily picked

        self.grid_shape = (
            image_width_in_pixels,
            image_width_in_pixels,
            image_width_in_pixels,
        )
        self.img_width = image_width_in_pixels
        self.img_shape = (image_width_in_pixels, image_width_in_pixels)
        self.noise_std = noise_std

        self.pixel_size = pixel_size
        self.voltage = voltage_in_kilovolts
        self.defocus_range = defocus_range
        self.astigmatism_range = astigmatism_range

        self.structures = atom_positions
        self.identities = identities
        self.b_factors = b_factors
        self.volumes = rendering._build_volumes(
            atom_positions,
            identities,
            self.b_factors,
            pixel_size,
            self.grid_shape,
        )

        self.latent_density = latent_density
        self.key = jax.random.PRNGKey(seed)

    def evaluate_latent_density(self, x):
        """
        # Evaluates the probability density function of the two Gaussian mixture model.

        Parameters
        -----------
        x:
            Location(s) to evaluate the probability density function.
        """
        return self.latent_density.evaluate_density(x)

    def sample_images(self, num_samples: int, shuffle=True):
        """
        Returns sampled_images

        Parameters
        ----------
        num_samples : int
            Number of samples
        shuffle: bool
            Whether to shuffle the images

        Returns
        -------
        images:
            Images
        structures:
            Corresponding images,
        latent_samples:
            Corresponding values of the latent.
        """
        latent_samples = self.latent_density.sample(num_samples, shuffle=shuffle)
        images, structures, poses, ctf_params = self.render_images_from_latent(
            latent_samples
        )
        return images, structures, poses, ctf_params, latent_samples

    def render_images_from_latent(
        self, latent_samples, poses=None, noise_std: Optional[float] = None
    ):
        if noise_std is None:
            noise_std = self.noise_std

        if poses is None:
            rotations = Rotation.random(len(latent_samples)).as_euler(seq="ZYZ")
            # rotations *= 180.0 / np.pi
            poses = np.zeros((len(latent_samples), 6))
            poses[:, 3:] += rotations
            poses = jnp.array(poses)

        volumes = self.volumes[latent_samples]
        structures = self.structures[latent_samples]
        images, ctfs = self.render_images_from_volumes(
            volumes, poses, noise_std=noise_std
        )
        # images, ctfs render_random_image(rotations, self.cjx_ensemble)
        return images, structures, poses, ctfs

    def render_projections_from_latent(self, latent_samples, poses=None):
        if poses is None:
            rotations = Rotation.random(len(latent_samples)).as_euler(seq="ZYZ")
            # rotations *= 180.0 / np.pi
            poses = np.zeros((len(latent_samples), 6))
            poses[:, 3:] += rotations
            poses = jnp.array(poses)

        volumes = self.volumes[latent_samples]
        structures = self.structures[latent_samples]
        images = rendering._render_projections_from_potential_grid(
            volumes, poses, self.img_shape, self.pixel_size, self.voltage
        )
        # images, ctfs render_random_image(rotations, self.cjx_ensemble)
        return images, structures, poses

    # def construct_structures(self, )

    def render_images_from_structures(
        self, atomic_structures, rotations, noise_std: Optional[float] = None
    ):
        """ """
        if noise_std is None:
            noise_std = self.noise_std

        N = len(atomic_structures)

        # TODO: replace with calls to jax random
        defocus = jnp.array(
            np.random.uniform(
                low=self.defocus_range[0], high=self.defocus_range[1], size=N
            )
        )
        astigmatism = jnp.array(
            np.random.uniform(
                low=self.astigmatism_range[0], high=self.astigmatism_range[1], size=N
            )
        )
        new_keys = jax.random.split(self.key, N + 1)

        images = rendering._render_noisy_images_from_atoms(
            atomic_structures,
            self.identities,
            rotations,
            noise_std,
            defocus,
            astigmatism,
            self.img_shape,
            self.pixel_size,
            self.voltage,
            new_keys[:N],
        )
        self.key = new_keys[-1]

        return images, (defocus, astigmatism)

    def render_images_from_volumes(self, volumes, rotations, noise_std=None):
        if noise_std is None:
            noise_std = self.noise_std

        N = len(volumes)

        # TODO: replace with calls to jax random
        defocus = jnp.array(
            np.random.uniform(
                low=self.defocus_range[0], high=self.defocus_range[1], size=N
            )
        )
        astigmatism = jnp.array(
            np.random.uniform(
                low=self.astigmatism_range[0], high=self.astigmatism_range[1], size=N
            )
        )

        new_keys = jax.random.split(self.key, N + 1)
        images = rendering._render_noisy_images_from_potential_grid(
            volumes,
            rotations,
            noise_std,
            defocus,
            astigmatism,
            self.img_shape,
            self.pixel_size,
            self.voltage,
            new_keys[:N],
        )
        self.key = new_keys[-1]
        return images, (defocus, astigmatism)

    def evaluate_pij_matrix(
        self,
        experimental_images,
        simulated_images,
        noise_std,
    ):
        """
        Computes the pij matrix from pre-rendered images.
        
        Parameters
        ----------
        experimental_images : jnp.ndarray
            Experimental images with shape (N_experimental, height, width)
        simulated_images : jnp.ndarray
            Simulated images with shape (N_simulated, height, width)
        noise_std : float or jnp.ndarray
            Noise standard deviation. Can be a scalar or array of shape (N_experimental,)
        
        Returns
        -------
        pij_matrix : jnp.ndarray
            Matrix where rows are experimental images and columns are simulated images.
            Shape is (N_experimental, N_simulated)
        """
        # Add new axes: (N_exp, H, W) -> (N_exp, 1, H, W) and (N_sim, H, W) -> (1, N_sim, H, W)
        experimental_images = jnp.expand_dims(experimental_images, axis=1)
        simulated_images = jnp.expand_dims(simulated_images, axis=0)
        
        # Compute squared difference and sum over spatial dimensions
        difference = jnp.sum(
            (experimental_images - simulated_images) ** 2, axis=(-2, -1)
        )
        
        # Handle noise_std as array or scalar
        if jnp.ndim(noise_std) > 0 and noise_std.shape[0] > 1:
            # noise_std has shape (N_exp,), reshape to (N_exp, 1) for broadcasting
            noise_std = jnp.reshape(noise_std, (-1, 1))
        
        # Compute log-likelihood
        loglikelihood = -1 * difference / (2 * noise_std**2)
        
        # Normalize by subtracting max value per row to avoid numerical overflow
        loglikelihood = loglikelihood - jnp.max(loglikelihood, axis=-1, keepdims=True)
        
        # Return normalized pij matrix
        return jnp.exp(loglikelihood)

    def evaluate_log_pij_matrix(
        self,
        experimental_images,
        predicted_atomic_structures,
        pose_as_euler_angle,
        noise_std: float,
    ):
        return _calculate_log_likelihood_of_atom_structures(
            predicted_atomic_structures,
            self.identities,
            self.b_factors,
            experimental_images,
            pose_as_euler_angle,
            noise_std,
            self.defocus,
            self.astigmatism,
            self.img_shape,
            self.pixel_size,
            self.voltage,
        )


class HSP90_Model(DiscreteClassModel):
    def __init__(
        self,
        latent_density=None,
        image_width_in_pixels: int = 128,
        pixel_size: float = 1.1,
        defocus_range=(5000.0, 10000.0),
        astigmatism_range=(0, 0),
        voltage_in_kilovolts=300.0,
        noise_std: float = 0.0,
        seed: int = 0,
    ):
        traj = _load_hsp90_traj()
        atom_positions, identities = get_atom_info_from_mdtraj(traj)

        super().__init__(
            atom_positions,
            identities,
            latent_density=latent_density,
            image_width_in_pixels=image_width_in_pixels,
            pixel_size=pixel_size,
            defocus_range=defocus_range,
            astigmatism_range=astigmatism_range,
            voltage_in_kilovolts=voltage_in_kilovolts,
            noise_std=noise_std,
            seed=seed,
        )
