"""
Loss Functions Module for EQ-KA-GCN

Provides Binary Cross Entropy (BCE) with class weights and Focal Loss for
handling severe class imbalance in the Tox21 SR-p53 toxicity prediction task
(~18:1 negative-to-positive ratio).

Focal Loss Reference:
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    https://arxiv.org/abs/1708.02002
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with severe class imbalance.

    Focal Loss down-weights easy, well-classified examples and focuses
    training on hard, misclassified examples. This is critical for the
    Tox21 SR-p53 task (~5.5% positives, ~18:1 imbalance ratio).

    Loss formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the model probability for the true class, alpha balances
    positive/negative classes, and gamma is the focusing parameter.

    Args:
        alpha (float): Weighting factor for the positive (toxic) class.
                       Recommended range: 0.25–0.75. Default: 0.25.
        gamma (float): Focusing parameter. Higher gamma = more focus on hard
                       examples. Recommended: 1.5–3.0. Default: 2.0.
        reduction (str): Loss reduction method ('mean', 'sum', 'none').
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes Focal Loss from raw logits and binary targets.

        Args:
            logits (torch.Tensor): Raw model logits, shape [batch_size, 1].
            targets (torch.Tensor): Binary targets (0 or 1), shape [batch_size, 1].

        Returns:
            torch.Tensor: Scalar focal loss value.
        """
        # Numerically stable BCE from logits: log(1 + exp(-logit)) for positive,
        # log(1 + exp(logit)) for negative
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # Convert logits to probabilities
        prob = torch.sigmoid(logits)

        # p_t: probability of the true class
        p_t = prob * targets + (1 - prob) * (1 - targets)

        # alpha_t: class-weighting factor for positive/negative classes
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Focal loss
        focal_loss = alpha_t * focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def get_loss_criterion(
    positive_class_weight: Optional[float] = None,
    use_focal_loss: bool = False,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """
    Constructs and returns the configured loss function.

    Priority order:
        1. FocalLoss (use_focal_loss=True)  ← Best for extreme class imbalance
        2. BCEWithLogitsLoss + pos_weight    ← Fallback class weighting
        3. BCEWithLogitsLoss (unweighted)   ← No imbalance handling

    Args:
        positive_class_weight (Optional[float]): Weight for positive class in BCE.
                                                  Only used when use_focal_loss=False.
        use_focal_loss (bool): If True, return FocalLoss instead of BCE.
        focal_alpha (float): Focal Loss alpha (positive class weight). Default: 0.25.
        focal_gamma (float): Focal Loss gamma (focusing parameter). Default: 2.0.

    Returns:
        nn.Module: Configured loss module (FocalLoss or BCEWithLogitsLoss).
    """
    if use_focal_loss:
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    if positive_class_weight is not None:
        pos_weight_tensor = torch.tensor([positive_class_weight], dtype=torch.float)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    return nn.BCEWithLogitsLoss()
