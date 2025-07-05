import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

config = yaml.safe_load(open("params.yaml"))

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNetEncoder(nn.Module):
    def __init__(self, in_ch, base_channels=16):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base_channels, base_channels*2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base_channels*2, base_channels*4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_channels*4, base_channels*8)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.bottleneck(self.pool3(x3))
        #print(f"Final shape: {x4.shape}")
        return [x1, x2, x3, x4]

class UNetDecoder(nn.Module):
    def __init__(self, out_ch, base_channels=16):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)  # sense output_padding
        self.dec1 = DoubleConv(base_channels*8, base_channels*4)
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec2 = DoubleConv(base_channels*4, base_channels*2)
        self.up3 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec3 = DoubleConv(base_channels*2, base_channels)
        self.out = nn.Conv2d(base_channels, out_ch, kernel_size=1)

    def forward(self, x_enc):
        x1, x2, x3, x4 = x_enc

        x = self.up1(x4)
        if x.size(2) != x3.size(2) or x.size(3) != x3.size(3):
            x3 = F.interpolate(x3, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        x = self.dec1(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        if x.size(2) != x2.size(2) or x.size(3) != x2.size(3):
            x2 = F.interpolate(x2, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up3(x)
        if x.size(2) != x1.size(2) or x.size(3) != x1.size(3):
            x1 = F.interpolate(x1, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        x = self.dec3(torch.cat([x, x1], dim=1))

        x = self.out(x)
        # Forcem la mida final a 300x300
        x = F.interpolate(x, size=tuple(config["dataset"]['cameras']['image_size']), mode='bilinear', align_corners=False)

        return x

class SingleCameraEncoder(nn.Module):
    def __init__(self, in_ch=3, base_channels=16):
        super().__init__()
        self.encoder = UNetEncoder(in_ch, base_channels)

    def forward(self, x):
        return self.encoder(x)

class BEVModel(nn.Module):
    def __init__(self, num_cameras=6, in_ch=3, out_ch=3, base_channels=16, fusion_method="mean"):
        super().__init__()
        self.encoders = nn.ModuleList([SingleCameraEncoder(in_ch, base_channels) for _ in range(num_cameras)])
        self.decoder = UNetDecoder(out_ch, base_channels)
        self.fusion_method = fusion_method
    
    def fuse_features(self, features, method="mean"):
        """
        Fusiona els features per nivell segons el mètode indicat.
        """
        fused = []

        for level in range(len(features[0])):
            level_feats = [feat[level] for feat in features]
            stacked = torch.stack(level_feats, dim=0)  # shape: [num_cams, B, C, H, W]

            if method == "mean":
                fused_level = torch.mean(stacked, dim=0)

            elif method == "max":
                fused_level, _ = torch.max(stacked, dim=0)

            elif method == "min":
                fused_level, _ = torch.min(stacked, dim=0)

            elif method == "concat":
                B, C, H, W = stacked[0].shape
                concat_feats = torch.cat(level_feats, dim=1)  # [B, C*num_cams, H, W]

                # Capa de reducció
                reduce_conv = nn.Conv2d(
                    in_channels=C * len(level_feats),
                    out_channels=C,
                    kernel_size=1
                ).to(concat_feats.device)

                fused_level = reduce_conv(concat_feats)

            else:
                raise ValueError(f"Unknown fusion method: {method}")

            fused.append(fused_level)

        return fused



    def forward(self, x):  # x: [B, C, D, W, H]
        B, C, D, W, H = x.shape
        x = x.to(torch.float32)
        features = []

        for i in range(C):
            cam_input = x[:, i]  # shape: [B, D, W, H]
            features.append(self.encoders[i](cam_input))

        # Fusion: promedio por nivel de features
        fused = self.fuse_features(features, method=self.fusion_method)


        output = self.decoder(fused)
        return output  # shape: [B, 3, 300, 300]
