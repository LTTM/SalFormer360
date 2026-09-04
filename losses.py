import math
import torch
import numpy as np
from torch import nn

def CC_Value(generated, GTh):
    x = generated
    y = GTh

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    loss_cc = torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2))
    )
    return loss_cc

def KL_loss(s_map, gt):
    """
    Compute Kullback-Leibler divergence (KL divergence) for batched tensors safely.

    Args:
        s_map (torch.Tensor): Predicted saliency maps of shape [b, c, h, w].
        gt (torch.Tensor): Ground truth saliency maps of shape [b, c, h, w].

    Returns:
        torch.Tensor: The KL divergence computed for each batch and channel, averaged over spatial dimensions.
    """
    # eps = 2.2204e-16  # Small constant to avoid log(0) and division by zero
    eps = torch.finfo(torch.float32).eps

    # Ensure non-zero normalization factors
    s_map_sum = s_map.sum(dim=(-2, -1), keepdim=True)
    gt_sum = gt.sum(dim=(-2, -1), keepdim=True)

    # Replace zeros in sums with a small value
    s_map_sum = s_map_sum + (s_map_sum == 0).float() * eps
    gt_sum = gt_sum + (gt_sum == 0).float() * eps

    # Normalize to probability distributions
    s_map = s_map / s_map_sum
    gt = gt / gt_sum

    # Compute KL divergence
    kl_div = gt * (torch.log(gt + eps) - torch.log(s_map + eps))

    # Sum over spatial dimensions (h, w) and average across batch and channels
    kl_div = kl_div.sum(dim=(-2, -1))  # Sum over spatial dimensions
    return kl_div.mean()  # Average over batch and channels


# S-MSE Loss
def sphere_mse(out, target, h, w):
    device = out.device  # Get the device of the input tensor
    weight = torch.zeros(
        1, 1, h, w, device=device
    )  # Initialize weight tensor on the same device
    theta_range = torch.linspace(
        0, math.pi, steps=h + 1, device=device
    )  # Move to device
    dtheta = math.pi / h
    dphi = 2 * math.pi / w
    for theta_idx in range(h):
        weight[:, :, theta_idx, :] = (
            dphi
            * (
                torch.sin(theta_range[theta_idx])
                + torch.sin(theta_range[theta_idx + 1])
            )
            / 2
            * dtheta
        )

    mse_loss = torch.sum((out - target) ** 2 * weight) / out.size(0)
    return mse_loss


BCE_loss = nn.BCEWithLogitsLoss()

def BCE_L(generated, GTh):
    loss_bce = BCE_loss(generated, GTh)
    return loss_bce


