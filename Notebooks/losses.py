import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- SSIM (simplified, window = 11x11 Gaussian approx with avg pooling) ----
def _ssim_map(x, y, C1=0.01**2, C2=0.03**2):
    # x,y in [0,1], shape (B,C,H,W)
    mu_x = F.avg_pool2d(x, 11, 1, 5)
    mu_y = F.avg_pool2d(y, 11, 1, 5)

    sigma_x  = F.avg_pool2d(x*x, 11, 1, 5) - mu_x*mu_x
    sigma_y  = F.avg_pool2d(y*y, 11, 1, 5) - mu_y*mu_y
    sigma_xy = F.avg_pool2d(x*y, 11, 1, 5) - mu_x*mu_y

    ssim_n = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    ssim_d = (mu_x*mu_x + mu_y*mu_y + C1) * (sigma_x + sigma_y + C2)
    ssim = ssim_n / (ssim_d + 1e-12)
    return ssim

def ssim_loss(x, y):
    ssim = _ssim_map(x, y)
    # convert similarity (1 best) to loss (0 best)
    return (1 - ssim).clamp(min=0, max=1).mean()

class L1_SSIM_Loss(nn.Module):
    def __init__(self, w_l1=0.8, w_ssim=0.2):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim

    def forward(self, x, xhat):
        l1 = F.l1_loss(xhat, x)
        s  = ssim_loss(xhat, x)
        return self.w_l1 * l1 + self.w_ssim * s

# ---- Top-k reconstruction error for image-level score ----
def topk_error(x, xhat, topk_frac=0.02):
    """
    x,xhat: (B,3,H,W) in [0,1]
    Returns:
      scores: (B,) mean of top-k per-pixel absolute errors (over channel avg)
      errmap: (B,1,H,W) absolute error map (channel-averaged)
    """
    err = (x - xhat).abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
    B, _, H, W = err.shape
    k = max(1, int(topk_frac * H * W))
    flat = err.view(B, -1)
    topk = torch.topk(flat, k, dim=1).values
    scores = topk.mean(dim=1)
    return scores, err
