import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import NoiseDiversityConfig, TrainingConfig
from model import NoiseEvaluator
from noise import DiverseNoiseGenerator


class AnomalyDetectorTrainer:
    """
    Trains a noise evaluation model following Algorithm 2.
    Works for both UAD and OCC — only the input data differs.
    """

    def __init__(
        self,
        training_config: TrainingConfig,
        noise_diversity_config: NoiseDiversityConfig,
        input_dim: int,
        device: torch.device = torch.device("cpu"),
        logger=None,

    ):
        self.training_config = training_config
        self.device = device

        self.logger = logger

        self.model = NoiseEvaluator(
            input_dim=input_dim,
            model_type=training_config.model_type,
        ).to(device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=training_config.lr,
            weight_decay=training_config.weight_decay,
            amsgrad=True,
        )
        """self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=training_config.lr_decay_epoch,
            gamma=training_config.lr_decay_factor,
        )"""
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[training_config.lr_decay_epoch],
            gamma=training_config.lr_decay_factor,
        )
        self.mse_loss = nn.MSELoss()
        self.noise_generator = DiverseNoiseGenerator(noise_diversity_config)

    def _log(self, line: str) -> None:
        if self.logger is not None:
            self.logger.write(line)
        else:
            print(line)
    def _train_step(self, batch_x: torch.Tensor) -> float:
        """One optimization step on a batch — Algorithm 2 inner loop."""
        batch_x = batch_x.to(self.device)

        # Generate K noisy versions per clean sample: shape (b*K, d)
        batch_x_hat = self.noise_generator.generate(batch_x)

        K = batch_x_hat.shape[0] // batch_x.shape[0]

        # Repeat clean batch K times to align: shape (b*K, d)
        batch_x_repeated = batch_x.repeat(K, 1)

        # Recover epsilon: epsilon = X_hat - X — Equation 8
        epsilon = batch_x_hat - batch_x_repeated

        # Loss on clean: predict 0 — Equation 9, first term
        pred_clean = self.model(batch_x)
        loss_clean = self.mse_loss(pred_clean, torch.zeros_like(pred_clean))

        # Loss on noisy: predict |epsilon| — Equation 9, second term
        pred_noisy = self.model(batch_x_hat)
        loss_noisy = self.mse_loss(pred_noisy, torch.abs(epsilon))

        loss = loss_clean + loss_noisy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def fit(self, X_train: torch.Tensor) -> nn.Module:
        """Train on X_train. For OCC, X_train should contain only normal samples.
        For UAD, X_train can include the full contaminated training set."""

        dataset = TensorDataset(X_train)
        dataloader = DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
        )

        self.model.train()
        for epoch in range(self.training_config.epochs):
            total_loss = 0.0
            for (batch_x,) in dataloader:
                total_loss += self._train_step(batch_x)

            self.scheduler.step()

            if (epoch + 1) % 50 == 0:
                avg = total_loss / len(dataloader)
                current_lr = self.optimizer.param_groups[0]["lr"]
                self._log(
                    f"Epoch [{epoch+1}/{self.training_config.epochs}] | "
                    f"Loss: {avg:.4f} | LR: {current_lr:.2e}"
                )

        return self.model