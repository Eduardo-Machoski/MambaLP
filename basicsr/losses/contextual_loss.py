import torch
import torch.nn.functional as F


def contextual_loss(x, y, h=0.5, eps=1e-5):
    """
    Contextual Loss (ECCV 2018)

    Args:
        x: (N, C, H, W)
        y: (N, C, H, W)

    Returns:
        scalar loss
    """

    N, C, H, W = x.shape

    # (N, HW, C)
    x = x.view(N, C, -1).permute(0, 2, 1).contiguous()
    y = y.view(N, C, -1).permute(0, 2, 1).contiguous()

    # Normalize features
    x = F.normalize(x, p=2, dim=2)
    y = F.normalize(y, p=2, dim=2)

    # Cosine similarity
    sim = torch.bmm(x, y.transpose(1, 2))

    # Cosine distance
    sim = torch.clamp(sim, -1.0, 1.0)
    d = 1.0 - sim

    # Relative distance
    d_min = d.min(dim=2, keepdim=True)[0]
    d_tilde = d / (d_min + eps)

    # Contextual weights
    w = torch.exp((1 - d_tilde) / h)

    # Normalize weights
    cx_ij = w / (w.sum(dim=2, keepdim=True) + eps)

    # Contextual similarity
    cx = cx_ij.max(dim=2)[0].mean(dim=1)

    # Loss
    loss = -torch.log(cx + eps)

    return loss.mean()
