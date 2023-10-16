import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def calculate_camera_direction(camera_position, visible_points):
    centroid = np.mean(visible_points, axis=0)
    direction = centroid - camera_position
    return direction / np.linalg.norm(direction)

# Generate random camera position and orientation
camera_position = (np.random.rand(3) * 4) - 2  # Random position between -2 and 2
camera_orientation = np.random.rand(3) * 360  # Random Euler angles in degrees

field_of_view = np.radians(60)

# Generate a random 3D point cloud normalized between -1 and 1
num_points = 2048
point_cloud = np.random.rand(num_points, 3) * 2 - 1

# Apply view frustum culling
visible_points = view_frustum_culling(point_cloud, camera_position, camera_orientation, field_of_view)

# Print the number of points before and after culling
print(f"Original points: {point_cloud.shape[0]}, Visible points: {visible_points.shape[0]}")

# Calculate camera direction towards the visible points
camera_direction = calculate_camera_direction(camera_position, visible_points)

# Visualize the point cloud and visible points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Original point cloud
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', marker='o', label='Original Points')

# Visible points after culling
ax.scatter(visible_points[:, 0], visible_points[:, 1], visible_points[:, 2], c='r', marker='x', label='Visible Points')

# Camera position
ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='g', marker='s', label='Camera Position')

# Add a line segment along the camera direction
line_points = np.vstack((camera_position, camera_position + camera_direction))
ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], c='k', label='Camera Direction')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()
