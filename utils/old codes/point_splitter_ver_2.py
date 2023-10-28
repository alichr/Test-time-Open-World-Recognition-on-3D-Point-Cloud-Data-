import torch
import numpy as np

class PointCloudProcessor:
    def __init__(self):
        pass
    
    @staticmethod
    def random_transform(point_cloud):
        # Apply random rotation
        rotation_matrix = torch.randn(point_cloud.shape[0], 3, 3)
        rotation_matrix, _ = torch.linalg.qr(rotation_matrix)
        rotated_points = torch.matmul(point_cloud, rotation_matrix)
        
        # Apply random split
        point_cloud_split = torch.zeros(rotated_points.shape[0], rotated_points.shape[1], rotated_points.shape[2])
        original_size = (rotated_points.shape[1], rotated_points.shape[2])
        for i in range(rotated_points.shape[0]):
            split_axis = np.random.randint(0, 3)
            part1 = rotated_points[i][rotated_points[i][:, split_axis] > 0]
            part2 = rotated_points[i][rotated_points[i][:, split_axis] <= 0]
            # make all points in part2 same as one point in part1
            part2 = torch.zeros_like(part2)
            point_cloud_split[i] = torch.cat((part1, part2), 0)
            # use interpolation to resize the point cloud to the original size
        
        return point_cloud_split