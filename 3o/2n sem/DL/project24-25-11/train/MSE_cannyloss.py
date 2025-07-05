import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

class MSECannyLoss(nn.Module):
    """
    Penalitza la pèrdua MSE  en els píxels detectats com a edges per Canny.
    """
    def __init__(self, edge_weight=2.0): #Provar diferents valors, per exemple 1.5, 2.0, 3.0. 
        """
        MSE a tota la imatge, pero l'error als egdes (del target) es multiplica pel `edge_weight`.
        """
        super(MSECannyLoss, self).__init__()
        self.edge_weight = edge_weight
        self.canny = kornia.filters.Canny()

    def forward(self, output, target):
        """
        :output: Predicció del model (B, C, H, W)
        :target: Ground truth (B, C, H, W)
        :return: Escalar amb la pèrdua MSE penalitzada
        """
        # 1. MSE per píxel amb reducció 'none'
        mse_map = F.mse_loss(output, target, reduction='none')  # (B, C, H, W)

        # 2. Canny sobre el target
        gray_target = kornia.color.rgb_to_grayscale(target)  # (B, 1, H, W)
        gray_target = (gray_target - gray_target.min()) / (gray_target.max() - gray_target.min() + 1e-8)
        edges, _ = self.canny(gray_target)  # (B, 1, H, W)

        # 3. Expandir la máscara als canals de color
        edge_mask = edges.expand_as(mse_map)  # (B, C, H, W)

        # 4. Incrementar penaliztació on hi ha bordes 
        weighted_mse = mse_map * (1.0 + self.edge_weight * edge_mask)

        return weighted_mse.mean()
