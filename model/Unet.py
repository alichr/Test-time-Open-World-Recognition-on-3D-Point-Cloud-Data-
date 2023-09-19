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
        self.encoder = ConvBlock(3, 32)
        self.center = ConvBlock(32, 64)
        self.decoder = ConvBlock(96, 32)
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x):
        enc = self.encoder(x)
        center = self.center(enc)
        dec = torch.cat([center, enc], dim=1)
        dec = self.decoder(dec)
        output = self.final_conv(dec)
        output = torch.sigmoid(output)
        return output



class UNetPlusPlusCondition(nn.Module):
    def __init__(self, conditional_dim):
        super(UNetPlusPlusCondition, self).__init__()
        self.conditional_dim = conditional_dim
        self.encoder = ConvBlock(1, 32)
        self.center = ConvBlock(32, 64)
        self.decoder = ConvBlock(96 + self.conditional_dim, 32)  # Updated input channels
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x, condition):
        enc = self.encoder(x)
        center = self.center(enc)
        condition_expanded = condition.view(condition.size(0), -1, 1, 1)  # Expand dimensions of condition
        condition_expanded = condition_expanded.expand(condition.size(0), -1, enc.size(2), enc.size(3))  # Expand spatial dimensions
  
        dec_input = torch.cat([center, enc, condition_expanded], dim=1)  # Concatenate condition
        dec = self.decoder(dec_input)
        output = self.final_conv(dec)
        return output

if __name__ == '__main__':
    conditional_dim = 3  # Define the dimension of your conditional vector
    model = UNetPlusPlusCondition(conditional_dim)
    x = torch.randn(1, 1, 254, 254)
    condition = torch.randn(1, conditional_dim)  # Example conditional vector
    y = model(x, condition)
    print(y.shape)
