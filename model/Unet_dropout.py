import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.5):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_prob)  # Add dropout layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        return x

class UNetPlusPlus(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(UNetPlusPlus, self).__init__()
        self.encoder = ConvBlock(3, 32, dropout_prob)
        self.center = ConvBlock(32, 64, dropout_prob)
        self.decoder = ConvBlock(96, 32, dropout_prob)
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)
        self.dropout = nn.Dropout2d(p=dropout_prob)  # Add dropout layer

    def forward(self, x):
        enc = self.encoder(x)
        center = self.center(enc)
        dec = torch.cat([center, enc], dim=1)
        dec = self.decoder(dec)
        dec = self.dropout(dec)  # Apply dropout
        output = self.final_conv(dec)
        output = torch.sigmoid(output)  
        return output
