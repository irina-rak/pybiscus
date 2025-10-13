import torch
import torch.nn as nn


class dice_3d(nn.Module):
    """Dice loss function."""

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        l_tot = 0
        for i in range(y_true.shape[1]):
            true_binary_map = torch.flatten(y_true[:, i, :, :, :])
            pred_map = torch.flatten(y_pred[:, i, :, :, :])
            dice_l = 1 - (
                2
                * torch.sum(torch.mul(true_binary_map, pred_map))
                / (torch.sum(torch.add(true_binary_map, pred_map)) + 1e-06)
            )
            l_tot += dice_l

        return l_tot


class WeightedDiceLoss3D(nn.Module):
    """Weighted dice loss function."""

    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, y_pred, y_true):
        """Compute the forward pass.

        Parameters:
        ----------
        y_true (torch.Tensor): the ground truth
        y_pred (torch.Tensor): the prediction
        """
        dice_loss = 0
        for k in range(1, y_true.shape[1]):
            true_binary_map = torch.flatten(y_true[:, k, :, :, :])
            pred_map = torch.flatten(y_pred[:, k, :, :, :])
            loc = 1 - (
                2
                * torch.sum(torch.mul(true_binary_map, pred_map))
                / (torch.sum(torch.add(true_binary_map, pred_map)) + 1e-6)
            )
            if self.class_weights is None:
                dice_loss += loc
            else:
                dice_loss += loc * self.class_weights[k]
        return dice_loss


class CrossEntropy(nn.Module):
    """Cross entropy loss function."""

    def __init__(self, class_weights) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, y_pred, y_true):
        """Compute the forward pass.

        Parameters:
        -----------
        y_true (torch.Tensor): the ground truth
        y_pred (torch.Tensor): the prediction
        """
        return self.loss(y_pred, y_true)


class CEDiceLoss(nn.Module):
    """Cross entropy + dice loss function."""

    def __init__(self, class_weights) -> None:
        super().__init__()
        self.ce = CrossEntropy(class_weights)
        self.wd = WeightedDiceLoss3D(class_weights)

    def forward(self, y_pred, y_true):
        """Compute the forward pass.

        Parameters:
        -----------
        y_true (torch.Tensor): the ground truth
        y_pred (torch.Tensor): the prediction
        """
        return self.ce(y_true, y_pred) + self.wd(y_true, y_pred)
