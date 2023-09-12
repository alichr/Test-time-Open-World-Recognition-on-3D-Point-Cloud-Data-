import torch
import torch.nn as nn

class Transformation(torch.nn.Module):
    def __init__(self, num_rotations=3):
        super(Transformation, self).__init__()
        self.num_rotations = num_rotations
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 9 * num_rotations)  # Output shape is (batch, 9 * num_rotations)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(9 * num_rotations)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.tanh(x)
        x = x.view(-1, self.num_rotations, 3, 3)
        return x
