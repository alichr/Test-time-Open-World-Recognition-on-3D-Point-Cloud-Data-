import torch
import torch.nn as nn

# Define the combined constraint loss
class CombinedConstraintLoss(nn.Module):
    def __init__(self, ortho_weight=1.0, scaling_weight=1.0, det_weight=1.0, num_rotations=3):
        super(CombinedConstraintLoss, self).__init__()
        self.ortho_weight = ortho_weight
        self.scaling_weight = scaling_weight
        self.det_weight = det_weight
        self.num_rotations = num_rotations

    def forward(self, Rs):
        device = Rs.device

        total_loss = 0.0

        for i in range(self.num_rotations):
            R = Rs[:, i, :, :]  # Select the i-th rotation matrix
            result = torch.matmul(R.transpose(1, 2), R)
            ortho_loss = torch.norm(result - torch.eye(3, device=device), p='fro')  # Frobenius norm

            scaling_factors = torch.diagonal(R, dim1=1, dim2=2)
            scaling_loss = torch.norm(scaling_factors - 1.0, p='fro')  # Frobenius norm

            det_loss = torch.abs(torch.det(R) - 1.0)

            total_loss += self.ortho_weight * ortho_loss + self.scaling_weight * scaling_loss + self.det_weight * det_loss

        total_loss /= self.num_rotations

        # Add orthogonal constraint among rotation matrices
        for i in range(self.num_rotations):
            for j in range(i+1, self.num_rotations):
                result = torch.matmul(Rs[:, i, :, :].transpose(1, 2), Rs[:, j, :, :])
                ortho_loss = torch.norm(result, p='fro')  # Frobenius norm
                total_loss += self.ortho_weight * ortho_loss

        return total_loss