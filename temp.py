# import numpy as np
# import os


# # # import numpy array from a file
# # def import_numpy_array(file):
# #     """
# #     Import a numpy array from a file.

# #     Parameters:
# #     - file: The file to import the array from.

# #     Returns:
# #     - array: The imported array.
# #     """
# #     array = np.load(file)
# #     return array

# # vec = import_numpy_array("Distance.npy")
# # # plot and show the array
# # import matplotlib.pyplot as plt
# # plt.plot(vec)
# # plt.xlabel('sample')
# # plt.ylabel('Distance')
# # plt.show()

# # load the array numoy
# text_data = np.load("Save_Mean_text_feature_image1000.npy",allow_pickle=True)
# # print the array

# # make a dictionary out of 
# # Define the dictionary with the given information
# novel_class_dic = {
#     26: ["n03337140", "n03742115", "n04550184"],
#     27: ["n02791124", "n03376595", "n04099969"],
#     28: ["n03179701"],
#     29: ["n03782006"],
#     30: ["n04239074"],
#     31: ["n03961711"],
#     32: ["n03201208", "n03982430"],
#     33: ["n04344873"],
#     34: ["0"],
#     35: ["n04344873"],
#     36: ["n04447861"]
# }

# # Function to find the key that matches the prediction
# def find_matching_key(prediction, category_dict):
#     for key, categories in category_dict.items():
#         if prediction in categories:
#             return key
#     return None

# pred = find_matching_key('0', novel_class_dic)
# print(pred)

import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

image = preprocess(Image.open("3D-to-2D-proj/guitar/test9.png")).unsqueeze(0)
text = tokenizer([
    "A depth map of a car depicts the diverse distances between the car's distinct features and the chosen perspective, contributing to a heightened sense of three-dimensionality within digital environments.",
    "A depth map of a guitar illustrates the varying distances of the guitar's surfaces and components from a designated viewpoint, serving to enhance its three-dimensional appearance in digital contexts.",
    "A depth map of a person represents the varying distances of different points on their body from a specific viewpoint, crucial for creating 3D realism in computer graphics and computer vision applications.", 
    "A depth map of a flower pot portrays a visual guide depicting the varying spatial gaps between the utilitarian structure, functional features, and practical design of the pot's construction and the viewer's designated angle, facilitating the generation of enhanced three-dimensional renderings.",
    "A depth map of a vase presents a visual representation showcasing the diverse spatial distances between the intricate curves, delicate contours, and refined details of the vase's silhouette and the observer's chosen standpoint, contributing to the creation of immersive three-dimensional visualizations."])
image_features = 0
with torch.no_grad(), torch.cuda.amp.autocast():
    for k in range(10):
        print(k)
        image_features = image_features + model.encode_image(preprocess(Image.open("3D-to-2D-proj/flowerpot/test" + str(k) + ".png")).unsqueeze(0))
    
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    print(text_features.shape)
    print(image_features.shape)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]