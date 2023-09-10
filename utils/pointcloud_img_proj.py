import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class PointCloudToDepthMap(nn.Module):
    def __init__(self, resolution, depth):
        super(PointCloudToDepthMap, self).__init__()
        self.resolution = resolution
        self.depth = depth

    def forward(self, point_cloud, transformation_matrices=None):
        batch_size, num_points, _ = point_cloud.size()

        if transformation_matrices is not None:
            assert transformation_matrices.size(0) == batch_size, "Number of transformation matrices should match batch size"
            assert transformation_matrices.size(1) == 3, "Transformation matrices should have shape (batch_size, 3, 4)"

            # Extend the point_cloud to [batch_size, num_points, 4] with homogeneous coordinate 1
            ones_column = torch.ones((batch_size, num_points, 1), device=point_cloud.device)
            point_cloud_h = torch.cat((point_cloud, ones_column), dim=2)

            # Expand the transformation matrices to (batch_size, 3, 4)
            # and apply to each point in the point cloud
            transformed_points_h = torch.matmul(point_cloud_h, transformation_matrices.permute(0, 2, 1))

            # Remove the homogeneous coordinate
            transformed_points = transformed_points_h[:, :, :3]

        else:
            transformed_points = point_cloud

        normalized_points = transformed_points / self.depth

        pixel_x = ((normalized_points[:, :, 0] + 1) / 2 * self.resolution).clamp(0, self.resolution - 1)
        pixel_y = ((1 - normalized_points[:, :, 1]) / 2 * self.resolution).clamp(0, self.resolution - 1)

        depth_map = torch.zeros((batch_size, self.resolution, self.resolution), dtype=torch.float32, device=point_cloud.device)

        for i in range(batch_size):
            for j in range(num_points):
                x = int(pixel_x[i, j])
                y = int(pixel_y[i, j])
                depth = normalized_points[i, j, 2]
                depth_map[i, y, x] = depth

        depth_map = F.interpolate(depth_map.unsqueeze(1), size=(self.resolution, self.resolution), mode='bilinear', align_corners=False)
        depth_map = depth_map.squeeze(1)

        return depth_map

# Example usage:
if __name__ == "__main__":
    resolution = 110
    depth = 0.5

    projection_layer = PointCloudToDepthMap(resolution, depth)

    # Generate a batch of random 3D point clouds with size [batch_size, 1024, 3] between -1 and 1
    batch_size = 4
    point_cloud = torch.rand((batch_size, 1024, 3), requires_grad=True) * 2 - 1  # Enable gradients

    # Define a batch of 3x4 transformation matrices (including rotation, scaling, and translation)
    transformation_matrices = torch.rand((batch_size, 3, 4), requires_grad=True)  # Enable gradients

    print(point_cloud.shape)

    # Project the transformed point clouds to depth maps
    depth_maps = projection_layer(point_cloud, transformation_matrices)

    # Normalize the depth maps for visualization
    depth_maps = (depth_maps - depth_maps.min()) / (depth_maps.max() - depth_maps.min()) * 255

    # Display the depth maps
    for i in range(batch_size):
        plt.figure()
        plt.imshow(depth_maps[i].cpu().detach().numpy(), cmap='viridis')
        plt.colorbar()
        plt.show()

    # Perform some operations and backpropagation for gradient computation
    loss = torch.sum(depth_maps)  # Just an example loss
    loss.backward()

    # Access gradients
    point_cloud_gradients = point_cloud.grad
    transformation_matrix_gradients = transformation_matrices.grad

    print("Point Cloud Gradients:", point_cloud_gradients)
    print("Transformation Matrix Gradients:", transformation_matrix_gradients)
