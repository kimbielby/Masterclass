import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        encoder4 = self.encoder4(self.pool(encoder3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(encoder4))

        # Decoder
        decoder4 = self.upconv4(bottleneck)
        decoder4 = self.decoder4(torch.cat([decoder4, encoder4], dim=1))

        decoder3 = self.upconv3(decoder4)
        decoder3 = self.decoder3(torch.cat([decoder3, encoder3], dim=1))

        decoder2 = self.upconv2(decoder3)
        decoder2 = self.decoder2(torch.cat([decoder2, encoder2], dim=1))

        decoder1 = self.upconv1(decoder2)
        decoder1 = self.decoder1(torch.cat([decoder1, encoder1], dim=1))

        return self.final_conv(decoder1)

def get_model(device, in_channels, out_channels):
    """
    Retrieves UNet model
    :param device: GPU or CPU
    :param in_channels: 3 or 5
    :param out_channels: 3
    :return: Model
    """
    model = UNet(in_channels=in_channels, out_channels=out_channels)  # 3, 3

    return model.to(device)







