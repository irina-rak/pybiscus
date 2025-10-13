from typing import Dict, Optional

import torch
import torch.nn as nn


class DiceScore(nn.Module):
    """Dice score metric."""

    higher_is_better: Optional[bool] = True

    def __init__(self, smooth: float = 1e-6):
        """Initialize DiceScore instance."""
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict:
        """Compute metric based on state.

        Arguments:
        ----------
        y_pred: torch.Tensor
            Predictions of the model.
        y_true: torch.Tensor
            Ground truth labels.

        Returns:
        --------
        scores: dict
            dict of dice scores for each class.
        """
        scores = {}
        pred_map = torch.argmax(y_pred, axis=1)
        for cls in range(y_true.shape[1]):
            true_binary_map = torch.flatten(y_true[:, cls, :, :, :])
            pred_binary_map = torch.flatten(
                torch.where(pred_map == cls, torch.tensor(1), torch.tensor(0))
            )
            intersection = torch.sum(torch.mul(true_binary_map, pred_binary_map))
            union = torch.sum(torch.add(true_binary_map, pred_binary_map))
            if union != 0:
                dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            elif intersection == 0:
                dice = torch.tensor(1.0)
            else:
                dice = torch.tensor(0.0)
            scores[f"dice_{cls}"] = dice
        return scores
