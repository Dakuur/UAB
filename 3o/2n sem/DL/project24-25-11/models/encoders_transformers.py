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

class TransformerEncoder(nn.Module):
    def __init__(self, latent_dim=128, num_heads=2, num_layers=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):  # x: (B, latent_dim, H, W)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C) = tokens
        x = self.transformer(x)  # (B, H*W, C)
        x = x.permute(0, 2, 1).view(B, C, H, W)  # reshape back to (B, C, H, W)
        return x

class BEVModel(nn.Module):
    def __init__(self, num_cameras=8, input_channels=3, latent_dim=128):
        super().__init__()
        self.encoders = nn.ModuleList([SingleCameraEncoder(input_channels=input_channels, latent_dim=latent_dim) for _ in range(num_cameras)])
        self.fusion = LatentFusion()
        self.transformer_decoder = TransformerEncoder(latent_dim=latent_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3, 1)  # output RGB
        )

    def forward(self, x):  # (B, 8, 3, H, W)
        B, C, D, H, W = x.shape
        latents = [self.encoders[i](x[:, i]) for i in range(C)]  # cada uno (B, latent_dim, H, W)
        fused = self.fusion(latents)  # (B, latent_dim, H, W)
        fused = self.transformer_decoder(fused)  # (B, latent_dim, H, W)
        decoded = self.decoder(fused)  # (B, 3, H, W)
        return decoded