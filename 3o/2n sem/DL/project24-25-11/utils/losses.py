import torch
import torch.nn as nn
import torch.nn.functional as F
import piq  # Per SSIM, PSNR
import lpips  # Per LPIPS

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.contiguous()
        targets = targets.contiguous()

        intersection = (preds * targets).sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (
            preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth
        )
        return 1 - dice.mean()


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        # debug
        #print(f"Preds values: {preds.min()} {preds.max()}")
        #print(f"Targets values: {targets.min()} {targets.max()}")
        preds = torch.clamp(preds, 0, 1)
        targets = torch.clamp(targets, 0, 1)
        #print(f"Preds values: {preds.min()} {preds.max()}")
        #print(f"Targets values: {targets.min()} {targets.max()}")
        return 1 - piq.ssim(preds, targets, data_range=1.0)


class PSNRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        preds = torch.clamp(preds, 0, 1)
        targets = torch.clamp(targets, 0, 1)
        return -piq.psnr(preds, targets, data_range=1.0)  # Negatiu perquè volem minimitzar

