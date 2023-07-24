import numpy as np
import sklearn
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
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    return kmeans

X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])    

cluster = clustering(X, n_clusters=2)
print(cluster.labels_)












