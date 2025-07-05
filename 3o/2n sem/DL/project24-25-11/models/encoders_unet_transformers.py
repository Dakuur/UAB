import torch
import torch.nn as nn
from models.encoders_unet import SingleCameraEncoder

class FeatureFusion(nn.Module):
    def __init__(self, method="mean"):
        super().__init__()
        self.method = method
        self.reduce_convs = nn.ModuleDict()

    def forward(self, features):
        fused = []
        num_levels = len(features[0])
        num_cams = len(features)
        B, C, H, W = features[0][0].shape

        for level in range(num_levels):
            level_feats = [feat[level] for feat in features]
            stacked = torch.stack(level_feats, dim=0)  # [num_cams, B, C, H, W]

            if self.method == "mean":
                fused_level = torch.mean(stacked, dim=0)

            elif self.method == "max":
                fused_level, _ = torch.max(stacked, dim=0)

            elif self.method == "min":
                fused_level, _ = torch.min(stacked, dim=0)

            elif self.method == "concat":
                concat_feats = torch.cat(level_feats, dim=1)  # [B, C*num_cams, H, W]
                key = f"level{level}"

                in_ch = concat_feats.shape[1]  # C * num_cams
                out_ch = C  # el nombre original de canals

                # Si no existeix o els canals no coincideixen, crea una nova capa
                if key not in self.reduce_convs or \
                   self.reduce_convs[key].in_channels != in_ch or \
                   self.reduce_convs[key].out_channels != out_ch:

                    self.reduce_convs[key] = nn.Conv2d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=1
                    ).to(concat_feats.device)

                reduce_conv = self.reduce_convs[key]
                fused_level = reduce_conv(concat_feats)

            else:
                raise ValueError(f"Unsupported fusion method: {self.method}")

            fused.append(fused_level)

        return fused

class TransformerDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, num_layers=2):
        super().__init__()

        self.project = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Transformer output processing
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Deconvolució (conv transpose) per anar pujant resolució
        self.deconv1 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)  # 37→75
        self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)  # 75→150

        # Upsample final a 300x300
        self.upsample = nn.Upsample(size=(300, 300), mode='bilinear', align_corners=False)

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape

        x_proj = self.project(x)
        x_flat = x_proj.flatten(2).permute(0, 2, 1)  # [B, HW, C]

        x_trans = self.transformer(x_flat)  # [B, HW, C]
        x_out = x_trans.permute(0, 2, 1).view(B, C, H, W)

        x_out = self.out(x_out)

        # Deconv 1 → Deconv 2 → Upsample
        x_out = self.deconv1(x_out)  # [B, out_ch, 75, 75]
        x_out = self.deconv2(x_out)  # [B, out_ch, 150, 150]
        x_out = self.upsample(x_out)  # [B, out_ch, 300, 300]

        return x_out

class BEVModel(nn.Module):
    def __init__(self, num_cameras=8, in_ch=3, out_ch=3, base_channels=16, fusion_method="mean"):
        super().__init__()
        self.encoders = nn.ModuleList([
            SingleCameraEncoder(in_ch, base_channels) for _ in range(num_cameras)
        ])
        self.fusion = FeatureFusion(method=fusion_method)

        if fusion_method != "concat":
            base_channels = 8* base_channels

        self.decoder = TransformerDecoder(in_channels=base_channels, out_channels=out_ch)

    def forward(self, x):  # x: [B, num_cams, in_ch, H, W]
        B, C, D, H, W = x.shape
        features = [self.encoders[i](x[:, i]) for i in range(C)]
        fused = self.fusion(features)
        return self.decoder(fused[-1])


