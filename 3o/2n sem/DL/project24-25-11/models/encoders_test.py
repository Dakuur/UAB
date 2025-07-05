import torch
import torch.nn as nn

class SingleCameraEncoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, latent_dim, 3, padding=1), nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)  # (B, latent_dim, H, W)

class LatentFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, latents):
        # shape de latents: (B, num_cameras, latent_dim)
        fused = torch.stack(latents, dim=0).sum(dim=0)  # suma los vectores latentes: (B, latent_dim)
        return fused

class BEVModel(nn.Module):
    def __init__(self, num_cameras=8, latent_dim=128):
        super().__init__()
        self.encoders = nn.ModuleList([SingleCameraEncoder() for _ in range(num_cameras)])
        self.fusion = LatentFusion()  # Sumar latentes de cámaras
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3, 1)  # output RGB values
        )

    def forward(self, x):  # x shape: (B, 8, 3, 300, 300)
        B, C, D, H, W = x.shape

        # 1. Aplicar el encoder a cada cámara
        latents = [self.encoders[i](x[:, i]) for i in range(C)]  # (B, latent_dim) por cada cámara

        # 2. Fusionar los latentes de las 8 cámaras
        fused = self.fusion(latents)  # (B, latent_dim)

        # 3. Decoder para generar la imagen final
        decoded = self.decoder(fused)  # (B, 3, H, W)
        return decoded