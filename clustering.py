import numpy as np
import torch
from foundation import clip
import torch.nn as nn
from utils.mv_utils_zs import Realistic_Projection
#from transformers import CLIPProcessor, CLIPModel
import open_clip
from PIL import Image

## define dataset and dataloader


## define clip model
def clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    return model, preprocess

## define projection
def projection():
    proj = Realistic_Projection()
    return proj

## define clustering
def clustering(X, n_clusters=10):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    return kmeans

# define SVD for creating subspace using pytroch
def SVD(X, n_components=10):
    U, S, V = torch.svd(X)
    return U[:, :n_components]


print(clip_model())
    















