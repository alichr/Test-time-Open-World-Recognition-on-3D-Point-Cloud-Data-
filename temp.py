import numpy as np
import os


# import numpy array from a file
def import_numpy_array(file):
    """
    Import a numpy array from a file.

    Parameters:
    - file: The file to import the array from.

    Returns:
    - array: The imported array.
    """
    array = np.load(file)
    return array

vec = import_numpy_array("Distance.npy")
# plot and show the array
import matplotlib.pyplot as plt
plt.plot(vec)
plt.show()


