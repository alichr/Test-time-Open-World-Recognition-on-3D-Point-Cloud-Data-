import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.cluster import KMeans
import argparse
from utils.dataloader import *
from pathlib import Path
from model.PointNet import PointNetfeat, feature_transform_regularizer

# Import any other required modules for the dataset and dataloader if needed
# (These parts are currently commented out)

import open_clip
from utils.mv_utils_zs import Realistic_Projection

# Check if a CUDA-enabled GPU is available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 

# define a function for pointnet
def pointnet():
    """
    Create and return a PointNet model.

    Returns:
    - model: The PointNet model.
    """
   # PointNetfeat(global_feat=True, feature_transform=opt.feature_transform).to(device)
    model = PointNetfeat(global_feat=True, feature_transform='store_true').to(device)
    return model


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
def main(opt):
    """
    Main function to execute the clustering process.

    Steps:
    1. Load CLIP model.
    2. Load Realistic Projection object.
    3. Generate random point cloud samples.
    4. Forward samples through the CLIP model to get feature vectors.
    5. Cluster the feature vectors using KMeans.
    """

    # deine data loader
    path = Path(opt.dataset_path)

    dataloader = DatasetGen(opt, root=path, fewshot=argument.fewshot)
    t = 0
    dataset = dataloader.get(t,'training')
    trainloader = dataset[t]['train']


    # Step 0: Load PointNet model
    model_3D = pointnet()
    model_3D = model_3D.to(device)
    model_3D.load_state_dict(torch.load(opt.model, map_location=torch.device('cpu')))

    # Step 1: Load CLIP model
    model_2D, preprocess = clip_model()

    # Step 2: Load Realistic Projection object
    proj = projection()
 
    
    # Step 3: Create 100 random samples using torch
    
    # Define a feature vectors size (100, 512)
    feature_vectors = np.zeros((len(trainloader.dataset), 512))

    # Step 4: Forward samples to the CLIP model
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            points, target = data['pointclouds'].to(device).float(), data['labels'].to(device)
            points, target = points.to(device), target.to(device)
            # Project samples to an image
            pc_prj = proj.get_img(points.unsqueeze(0))
            pc_img = torch.nn.functional.interpolate(pc_prj, size=(224, 224), mode='bilinear', align_corners=True)  
            pc_img = pc_img.to(device)
            # Forward samples to the CLIP model
            pc_img = model_2D.encode_image(pc_img)
            pc_img = pc_img.cpu().numpy()
            pc_img_avg = np.mean(pc_img, axis=0)        
            # Save feature vectors
            feature_vectors[i,:] = pc_img_avg
            print(i)
    
    # Step 5: Cluster samples
    kmeans = clustering(feature_vectors, n_clusters=10)

    # number of clusters

    # Step6: seperate the clusters into different groups and apply SVD to each group and save each subsapce in a subsapce folder
    for i in range(kmeans.n_clusters):
        idx = np.where(kmeans.labels_ == i)[0]  # Use [0] to access the indices array from tuple
        cluster_feature_vectors = feature_vectors[idx]
        print(cluster_feature_vectors.shape)
        U, S, V = SVD_numpy(cluster_feature_vectors, n_components=3)
        np.save('subspace/subspace' + str(i) + '.npy', U)
        print(i)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--num_points', type=int, default=1024, help='number of points in each input point cloud')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder to save results')
    parser.add_argument('--model', type=str, default='cls/3D_model_249.pth', help='path to load a pre-trained model')
    parser.add_argument('--feature_transform', action='store_true', help='use feature transform')
    parser.add_argument('--manualSeed', type=int, default = 42, help='random seed')
    parser.add_argument('--dataset_path', type=str, default= 'dataset/modelnet_scanobjectnn/', help="dataset path")
    parser.add_argument('--ntasks', type=str, default= '1', help="number of tasks")
    parser.add_argument('--nclasses', type=str, default= '26', help="number of classes")
    parser.add_argument('--task', type=str, default= '0', help="task number")
    parser.add_argument('--num_samples', type=str, default= '0', help="number of samples per class")


    opt = parser.parse_args()

    main(opt)
    print("Done!")

 
