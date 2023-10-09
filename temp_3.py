import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def view_frustum_culling(point_cloud, camera_position, camera_orientation, field_of_view):
    rotation_matrix = get_rotation_matrix_from_angles(camera_orientation)
    translated_points = point_cloud - camera_position
    rotated_points = np.dot(translated_points, rotation_matrix.T)
    projected_points = rotated_points[:, :2] / rotated_points[:, 2].reshape(-1, 1)
    x_bound = np.tan(field_of_view / 2)
    y_bound = x_bound
    valid_indices = np.logical_and(
        np.logical_and(-x_bound <= projected_points[:, 0], projected_points[:, 0] <= x_bound),
        np.logical_and(-y_bound <= projected_points[:, 1], projected_points[:, 1] <= y_bound)
    )
    return point_cloud[valid_indices]

def get_rotation_matrix_from_angles(angles):
    return R.from_euler('xyz', angles, degrees=True).as_matrix()

# Define camera parameters
camera_position = np.array([0, 0, 5])
camera_orientation = np.array([0, 0, 0])
field_of_view = np.radians(60)

# Generate a random 3D point cloud normalized between -1 and 1
num_points = 2048
point_cloud = np.random.rand(num_points, 3) * 2 - 1

# Apply view frustum culling
visible_points = view_frustum_culling(point_cloud, camera_position, camera_orientation, field_of_view)

# Visualize input point cloud
plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', s=5, label='Input Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Input Point Cloud')
plt.legend()
plt.show()

# Visualize output point cloud
plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
ax.scatter(visible_points[:, 0], visible_points[:, 1], visible_points[:, 2], c='g', s=5, label='Visible Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Visible Point Cloud')
plt.legend()
plt.show()
