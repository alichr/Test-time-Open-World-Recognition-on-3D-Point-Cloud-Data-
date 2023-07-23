import numpy as np
#import sklearn
import torch
from foundation import clip
import torch.nn as nn
from utils.mv_utils_zs import Realistic_Projection
#from transformers import CLIPProcessor, CLIPModel
import open_clip


## define dataset and dataloader



## define model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
#model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
print(model)



# print(model.encode_image())

    


