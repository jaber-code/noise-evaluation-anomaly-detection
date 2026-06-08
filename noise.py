import math
import random
import torch

import config
from enums import NoiseType


class DiverseNoiseGenerator:

    _NO_SIGMA_SCALE = {NoiseType.BERNOULLI, NoiseType.SALT_PEPPER}

    def __init__(self, noise_diversity_configs: config.NoiseDiversityConfig):
        self.noise_diversity_configs = noise_diversity_configs
        self.m = noise_diversity_configs.m
        self.noise_dist_dict = noise_diversity_configs.noise_dist_configs
        self.used_noise_distributions = noise_diversity_configs.used_noise_distributions

    def _sample_noise(self, b: int, chunk_size: int, noise_type: NoiseType,
                      X_chunk: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Returns unit noise (mean=0, std≈1). Caller handles sigma scaling."""

        if noise_type == NoiseType.GAUSSIAN:
            return torch.randn(b, chunk_size, device=device)

        elif noise_type == NoiseType.LAPLACE:
            return torch.distributions.Laplace(
                torch.tensor(0.0, device=device),
                torch.tensor(1.0 / math.sqrt(2.0), device=device)
            ).sample((b, chunk_size))

        elif noise_type == NoiseType.UNIFORM:
            return (torch.rand(b, chunk_size, device=device) * 2.0 - 1.0) * math.sqrt(3.0)

        elif noise_type == NoiseType.RAYLEIGH:
            s = math.sqrt(2.0 / (4.0 - math.pi))
            offset = -math.sqrt(math.pi / (4.0 - math.pi))
            rayleigh = s * torch.sqrt(
                torch.randn(b, chunk_size, device=device) ** 2 +
                torch.randn(b, chunk_size, device=device) ** 2
            )
            return rayleigh + offset

        elif noise_type == NoiseType.GAMMA:
            beta = 1.0  # Table 3: beta=1 or beta=3
            alpha = 1.0 / math.sqrt(beta)   # Table 3: alpha = sqrt(1/beta) * sigma_hat, sigma_hat=1
            offset = -math.sqrt(beta)       # Table 3: offset = -sqrt(beta) * sigma_hat
            return torch.distributions.Gamma(
                torch.tensor(alpha, device=device),
                torch.tensor(beta, device=device)
            ).sample((b, chunk_size)) + offset

        elif noise_type == NoiseType.POISSON:
            # Table 3: lambda = sigma_hat = 1 for unit noise, offset = -1
            return torch.distributions.Poisson(
                torch.tensor(1.0, device=device)
            ).sample((b, chunk_size)) - 1.0

        elif noise_type == NoiseType.BERNOULLI:
            # Paper: flip sign of element → epsilon = -X - X = -2X
            return -2.0 * X_chunk

        elif noise_type == NoiseType.SALT_PEPPER:
            # Paper: replace with batch max or min
            batch_max = X_chunk.max(dim=0, keepdim=True)[0]
            batch_min = X_chunk.min(dim=0, keepdim=True)[0]
            use_max = torch.rand_like(X_chunk) > 0.5
            target = torch.where(use_max, batch_max, batch_min)
            return target - X_chunk

        else:
            raise NotImplementedError(f"Noise type {noise_type} not implemented.")

    def _generate_single_distribution(
        self, X: torch.Tensor, noise_type: NoiseType, noise_ratio: float
    ) -> torch.Tensor:
        """Generates one X_hat = X + epsilon with the given noise_type and noise_ratio."""
        b, d = X.shape
        device = X.device

        noise_cfg = self.noise_dist_dict[noise_type]
        sigma_max = noise_cfg.sigma_max

        if self.m > d:
            raise ValueError(f"m ({self.m}) must not exceed feature dimension d ({d})")

        epsilon = torch.zeros(b, d, device=device)

        # Build m equal-width sigma intervals: Delta = [0, sigma_max/m, ..., sigma_max]
        # Algorithm 1, step 1 — not needed for sigma-free types (BERNOULLI, SALT_PEPPER)
        delta = None
        if noise_type not in DiverseNoiseGenerator._NO_SIGMA_SCALE:
            assert sigma_max is not None
            delta = torch.linspace(0, sigma_max, self.m + 1, device=device)

        for i in range(self.m):
            start_idx = i * (d // self.m)
            end_idx = d if i == self.m - 1 else (i + 1) * (d // self.m)
            current_chunk_size = end_idx - start_idx

            # Sample unit noise for this chunk
            unit_noise = self._sample_noise(
                b, current_chunk_size, noise_type,
                X[:, start_idx:end_idx], device
            )

            # Scale by sigma_hat: σ̂ ← Uniform(∆[i], ∆[i+1])  — Algorithm 1, steps 6-7
            if noise_type in DiverseNoiseGenerator._NO_SIGMA_SCALE:
                noise_chunk = unit_noise
            else:
                assert delta is not None
                sigma_hat = torch.empty(b, 1, device=device).uniform_(
                    delta[i].item(), delta[i + 1].item()
                )
                noise_chunk = unit_noise * sigma_hat  # broadcasts to (b, chunk_size)

            # Apply noise_ratio: zero out (1 - noise_ratio) fraction of features within chunk
            # The post-loop shuffle randomizes which positions end up corrupted.
            if noise_ratio < 1.0:
                n_noisy = max(1, int(noise_ratio * current_chunk_size))
                noise_chunk[:, n_noisy:] = 0

            epsilon[:, start_idx:end_idx] = noise_chunk

        # Shuffle noise positions across full feature vector — Algorithm 1, step 10
        shuffle_idx = torch.rand(b, d, device=device).argsort(dim=1)
        epsilon = torch.gather(epsilon, 1, shuffle_idx)

        return X + epsilon

    def generate(self, X: torch.Tensor) -> torch.Tensor:
        """
        Paper protocol: pick one noise type, then generate one noisy instance
        per noise_ratio in that distribution's list.

        With noise_ratios=[0.5, 0.8, 1.0], emits 3 noisy instances per clean sample.
        Output shape: (b * K, d) where K = len(noise_ratios) of the chosen distribution.
        """
        noisy_samples_list = []
        for noise_type in self.used_noise_distributions:
            ratios = self.noise_dist_dict[noise_type].noise_ratios
            for r in ratios:
                X_hat = self._generate_single_distribution(X, noise_type, r)
                noisy_samples_list.append(X_hat)

        return torch.cat(noisy_samples_list, dim=0)