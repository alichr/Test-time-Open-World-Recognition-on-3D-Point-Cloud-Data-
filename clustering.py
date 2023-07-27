import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.cluster import KMeans

# Import any other required modules for the dataset and dataloader if needed
# (These parts are currently commented out)

import open_clip
from utils.mv_utils_zs import Realistic_Projection

# Check if a CUDA-enabled GPU is available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Define CLIP model
def clip_model():
    """
    Create and return a CLIP model with a specified architecture and pre-trained weights.

    Returns:
    - model: The CLIP model.
    - preprocess: The preprocessing function to prepare images for the model.
    """
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    return model, preprocess

## Define projection
def projection():
    """
    Create and return a Realistic Projection object.

    Returns:
    - proj: Realistic Projection object.
    """
    proj = Realistic_Projection()
    return proj

## Define clustering
def clustering(X, n_clusters=10):
    """
    Cluster data X into n_clusters groups using KMeans algorithm.

    Parameters:
    - X: The data to be clustered.
    - n_clusters: The number of clusters to create.

    Returns:
    - kmeans: KMeans model fitted to the data X.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    return kmeans

# Define SVD for creating a subspace using numpy
def SVD_numpy(X, n_components=3):
    """
    Perform Singular Value Decomposition (SVD) on data X using numpy.

    Parameters:
    - X: The data for SVD.
    - n_components: The number of top singular values and vectors to keep.

    Returns:
    - U: Left singular vectors of X.
    - S: Singular values of X.
    - V: Right singular vectors of X.
    """
    U, S, V = np.linalg.svd(X, full_matrices=False)
    return U[:, :n_components], S, V

# Define SVD for creating a subspace using PyTorch
def SVD(X, n_components=10):
    """
    Perform Singular Value Decomposition (SVD) on data X using PyTorch.

    Parameters:
    - X: The data for SVD.
    - n_components: The number of top singular values and vectors to keep.

    Returns:
    - U: Left singular vectors of X.
    - S: Singular values of X.
    - V: Right singular vectors of X.
    """
    U, S, V = torch.svd(X)
    return U[:, :n_components], S, V


## Define main function
def main():
    """
    Main function to execute the clustering process.

    Steps:
    1. Load CLIP model.
    2. Load Realistic Projection object.
    3. Generate random point cloud samples.
    4. Forward samples through the CLIP model to get feature vectors.
    5. Cluster the feature vectors using KMeans.
    """
    # Step 1: Load CLIP model
    model, preprocess = clip_model()

    # Step 2: Load Realistic Projection object
    proj = projection()

    # Step 3: Create 100 random samples using torch
    pc = torch.rand(100, 1024, 3)
    # Define a feature vectors size (100, 512)
    feature_vectors = np.zeros((100, 512))

    # Step 4: Forward samples to the CLIP model
    with torch.no_grad():
        for i in range(100):
            # Project samples to an image
            pc_prj = proj.get_img(pc[i,:,:].unsqueeze(0))
            pc_img = torch.nn.functional.interpolate(pc_prj, size=(224, 224), mode='bilinear', align_corners=True)  
            pc_img = pc_img.to(device)
            # Forward samples to the CLIP model
            pc_img = model.encode_image(pc_img)
            pc_img = pc_img.cpu().numpy()
            pc_img_avg = np.mean(pc_img, axis=0)        
            # Save feature vectors
            feature_vectors[i,:] = pc_img_avg
            print(i)
    
    # Step 5: Cluster samples
    kmeans = clustering(feature_vectors, n_clusters=10)

    # Step6: seperate the clusters into different groups and apply SVD to each group and save each subsapce in a subsapce folder
    for i in range(10):
        idx = np.where(kmeans.labels_ == i)[0]  # Use [0] to access the indices array from tuple
        cluster_feature_vectors = feature_vectors[idx]
        print(cluster_feature_vectors.shape)
        U, S, V = SVD_numpy(cluster_feature_vectors, n_components=3)
        np.save('subspace/subspace' + str(i) + '.npy', U)
        print(i)

    # step 7: min absoulte error of extracted subsapce and original subspace
    # load the original subspace
    original_subspace = np.load('subspace/subspace.npy')
    # load the extracted subspace
    extracted_subspace = np.load('subspace/subspace0.npy')
    # calculate the min absoulte error
    min_abs_error = np.min(np.abs(original_subspace - extracted_subspace))
    print(min_abs_error)



  


if __name__ == "__main__":
    main()
    print("Done!")

 
