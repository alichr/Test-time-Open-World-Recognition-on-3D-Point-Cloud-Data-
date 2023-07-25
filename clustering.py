import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.cluster import KMeans

# Import any other required modules for the dataset and dataloader if needed
# (These parts are currently commented out)

import open_clip
from utils.mv_utils_zs import Realistic_Projection

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Define CLIP model
def clip_model():
    # Create a CLIP model with a specified architecture and pre-trained weights
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    return model, preprocess

## Define projection
def projection():
    # Create a Realistic Projection object
    proj = Realistic_Projection()
    return proj

## Define clustering
def clustering(X, n_clusters=10):
    # Use KMeans algorithm to cluster data X into n_clusters groups
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    return kmeans

# Define SVD for creating subspace using PyTorch
def SVD(X, n_components=10):
    # Perform Singular Value Decomposition (SVD) on data X using PyTorch
    U, S, V = torch.svd(X)
    return U[:, :n_components]


## Define main function
def main():

    # Load CLIP model
    model, preprocess = clip_model()

    # Load Realistic Projection object
    proj = projection()

    # create 100 random samples using torch
    pc = torch.rand(100, 1024, 3)

    # project samples to image
    pc_prj = proj.get_img(pc)
    pc_img = torch.nn.functional.interpolate(pc_prj, size=(224, 224), mode='bilinear', align_corners=True)  

    # forwad samples to clip model
    with torch.no_grad():
        pc_img = pc_img.to(device)
        pc_img = model.encode_image(pc_img)
        pc_img = pc_img.cpu().numpy()
        print(pc_img.shape)

    # cluster samples
    kmeans = clustering(pc_img, n_clusters=10)




if __name__ == "__main__":
    main()














