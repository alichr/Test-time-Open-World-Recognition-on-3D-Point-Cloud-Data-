import numpy as np
import sklearn
import torch
from foundation import clip
import torch.nn as nn
from utils.mv_utils_zs import Realistic_Projection



## define dataset and dataloader



## define model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)



print(model.encode_image())

    


