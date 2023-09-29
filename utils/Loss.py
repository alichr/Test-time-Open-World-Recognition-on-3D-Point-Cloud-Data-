import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

# clip model loss  
class ClipLoss(nn.Module):
    def __init__(self, device, t : float = 0.1):
        super(ClipLoss, self).__init__()
        self.device = device
        self.t = nn.Parameter(torch.ones([])* np.log(1/t)).exp().to(device)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, feature_embeddings, semantic_embeddings):
        feature_embeddings = F.normalize(feature_embeddings)
        semantic_embeddings = F.normalize(semantic_embeddings)

        # scaled pairwise cosine similarities [n, n]
        logits = feature_embeddings @ semantic_embeddings.t() * self.t

        # symmetric loss function
        batch_size = feature_embeddings.shape[0]
        labels = torch.arange(batch_size).to(self.device)

        loss_features = self.loss(input=logits, target=labels)
        loss_semantics = self.loss(input=logits.T, target=labels)
        loss = (loss_features + loss_semantics) / 2
        # print("loss_it", loss_features, loss_semantics)

        return loss