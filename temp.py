import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class PointCloudSpliter:
    def __init__(self):
        pass

    def random_rotation(self, point_cloud):
        rotation_matrix = torch.randn(3, 3)
        rotation_matrix, _ = torch.linalg.qr(rotation_matrix)
        rotated_points = torch.matmul(point_cloud, rotation_matrix)
        return rotated_points

    def random_split(self, point_cloud):
        split_axis = np.random.randint(0, 3)
        split_value = torch.median(point_cloud[:, :, split_axis])
        part1 = point_cloud[point_cloud[:, :, split_axis] <= split_value]
        part2 = point_cloud[point_cloud[:, :, split_axis] > split_value]
        return part1, part2

    def interpolate_to_original(self, part, original_size):
        interpolated_part = F.interpolate(part.unsqueeze(0).unsqueeze(0), size=original_size, mode='bilinear')
        return interpolated_part.squeeze(0).squeeze(0)

    def process_point_cloud(self, point_cloud):
        rotated_point_cloud = self.random_rotation(point_cloud)
        part1, part2 = self.random_split(rotated_point_cloud)
        chosen_part = part1 if torch.rand(1) > 0.5 else part2
        original_size = (point_cloud.shape[1], point_cloud.shape[2])
        interpolated_part = self.interpolate_to_original(chosen_part, original_size)
        return interpolated_part


if __name__ == '__main__':
    # Example of how to use the class
    spliter = PointCloudSpliter()

    # Generate random point cloud dataset
    batch_size = 1
    point_cloud = torch.randn(batch_size, 2048, 3)

    # Process the point cloud
    output_part = spliter.process_point_cloud(point_cloud)
    print(output_part.shape)


    # Visualize the point clouds
    def visualize_point_cloud(point_cloud, title):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, c='b', marker='.')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        plt.show()

    visualize_point_cloud(point_cloud[0], 'Original Point Cloud')
    visualize_point_cloud(output_part, 'Processed Point Cloud')
