import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class UNetPlusPlus(nn.Module):
    def __init__(self):
        super(UNetPlusPlus, self).__init__()
        self.encoder1 = ConvBlock(1, 32)
        self.encoder2 = ConvBlock(32, 64)
        self.encoder3 = ConvBlock(64, 128)
        self.encoder4 = ConvBlock(128, 256)
        
        self.center = ConvBlock(256, 512)
        
        self.decoder4 = ConvBlock(768, 256)
        self.decoder3 = ConvBlock(384, 128)
        self.decoder2 = ConvBlock(192, 64)
        self.decoder1 = ConvBlock(96, 32)
        
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        center = self.center(enc4)
        
        dec4 = torch.cat([center, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = torch.cat([dec4, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = torch.cat([dec3, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = torch.cat([dec2, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        output = self.final_conv(dec1)
        return output


if __name__ == '__main__':
    # Initialize the lighter model
    model = UNetPlusPlus()
    print(model)

    # Test the model with a random input
    random_input = torch.randn(1, 1, 224, 224)
    output = model(random_input)

    # Print the output shape
    print(output.shape)  # Should be torch.Size([1, 3, 224, 224])
