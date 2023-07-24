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
# generate 100 random samples with size 1024 x 3
X = np.random.rand(100, 1024, 3)


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

    # create 100 random samples using torch
    X = torch.rand(100, 1024, 3)

    # Load CLIP model
    model, preprocess = clip_model()

    # Load Realistic Projection object
    proj = projection()

    image_prject = proj.get_img(X[:2])

    print(image_prject.shape)
    

    fuck


    # Preprocess data X
    X = preprocess(X)

    # Create subspace using SVD
    X = SVD(X)

    # Cluster data X into n_clusters groups
    n_clusters = 10
    kmeans = clustering(X, n_clusters)

    # Save the cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Save the cluster labels
    cluster_labels = kmeans.labels_

    # Save the cluster inertia
    cluster_inertia = kmeans.inertia_

    # Save the cluster sizes
    cluster_sizes = np.bincount(cluster_labels)

if __name__ == "__main__":
    main()














