"""
Intensity-weighted loss — referenced on Slide 2/3 of our PPT.

Plain MSE trains a model to predict the CONDITIONAL MEAN, which for
precipitation nowcasting means the model learns to blur out intense,
localized storm cores (hail cores, cloudburst cells) in favor of a
smoother, safer-looking average field. This is a well-documented
failure mode in nowcasting literature (see docs/references.md,
arXiv:2601.06137).

Fix: weight each pixel's squared error by a function of its TARGET
intensity, so the loss penalizes errors on intense storm cores far
more heavily than errors on light/background rain. This keeps sharp,
high-dBZ features from being regressed away during training.
"""

import torch
import torch.nn as nn


class IntensityWeightedLoss(nn.Module):
    """
    Weighted MSE: weight(y) = 1 + alpha * y^gamma

    - alpha controls how much extra weight intense pixels get.
    - gamma > 1 makes the weighting even steeper for the most extreme
      values (hail cores, cloudburst rain rates), matching how
      operational verification (e.g. CSI at high thresholds) cares
      disproportionately about getting extremes right.

    Args:
        alpha: weighting strength (default 4.0 — tune against a
            validation set; higher = more aggressive on storm cores).
        gamma: exponent on target intensity (default 1.5).
        max_val: expected max of the target variable (e.g. ~150 mm/hr
            for extreme rain rate), used to normalize y before
            exponentiating so alpha/gamma stay interpretable across
            variables with different units.
    """

    def __init__(self, alpha: float = 4.0, gamma: float = 1.5, max_val: float = 150.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.max_val = max_val

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_norm = torch.clamp(target / self.max_val, min=0.0, max=1.0)
        weight = 1.0 + self.alpha * torch.pow(y_norm, self.gamma)
        se = (pred - target) ** 2
        return (weight * se).mean()


if __name__ == "__main__":
    loss_fn = IntensityWeightedLoss()
    pred = torch.rand(4, 1, 64, 64) * 50
    target = torch.rand(4, 1, 64, 64) * 150
    print("Loss:", loss_fn(pred, target).item())
