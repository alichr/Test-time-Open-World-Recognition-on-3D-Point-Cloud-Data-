import numpy as np
import os


# # import numpy array from a file
# def import_numpy_array(file):
#     """
#     Import a numpy array from a file.

#     Parameters:
#     - file: The file to import the array from.

#     Returns:
#     - array: The imported array.
#     """
#     array = np.load(file)
#     return array

# vec = import_numpy_array("Distance.npy")
# # plot and show the array
# import matplotlib.pyplot as plt
# plt.plot(vec)
# plt.xlabel('sample')
# plt.ylabel('Distance')
# plt.show()

# load the array numoy
text_data = np.load("Save_Mean_text_feature_image1000.npy",allow_pickle=True)
# print the array

# make a dictionary out of 
# Define the dictionary with the given information
novel_class_dic = {
    26: ["n03337140", "n03742115", "n04550184"],
    27: ["n02791124", "n03376595", "n04099969"],
    28: ["n03179701"],
    29: ["n03782006"],
    30: ["n04239074"],
    31: ["n03961711"],
    32: ["n03201208", "n03982430"],
    33: ["n04344873"],
    34: ["0"],
    35: ["n04344873"],
    36: ["n04447861"]
}

# Function to find the key that matches the prediction
def find_matching_key(prediction, category_dict):
    for key, categories in category_dict.items():
        if prediction in categories:
            return key
    return None

pred = find_matching_key('0', novel_class_dic)
print(pred)