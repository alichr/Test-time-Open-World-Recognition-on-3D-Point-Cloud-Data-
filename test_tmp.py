import cv2
import numpy as np
import matplotlib.pyplot as plt
import os



# load a png image
img = cv2.imread('depth_map.png', cv2.IMREAD_UNCHANGED)
depth_map = img.astype(np.uint8)
print(depth_map)

# visualize the depth map
plt.imshow(depth_map)
plt.show()


depth_map_reverse = (255 - depth_map).astype(np.uint8)
plt.imshow(depth_map_reverse)
plt.show()
print(depth_map_reverse.shape)

# create a mask
mask = np.zeros((depth_map.shape[0], depth_map.shape[1], depth_map.shape[2]), dtype=np.uint8)
mask[depth_map_reverse > 0] = 255


# visualize the mask
plt.imshow(mask)
plt.show()

# replace the background with random values
background = np.random.randint(0, 255, (depth_map.shape[0], depth_map.shape[1], depth_map.shape[2]), dtype=np.uint8)
# add values from the depth map for the foreground
background[mask == 255] = depth_map[mask == 255]

plt.imshow(background)
plt.show()