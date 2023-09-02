import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import random
import pickle
import pdb
import json
import csv
import pandas as pd
from torch.utils.data import DataLoader


import os
import pandas as pd
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torchvision.transforms import Resize

class MiniImageNet(Dataset):
    def __init__(self, img_list, img_dir, transform=None, target_transform=None, image_format="RGB"):
        with open(img_list, 'r') as f:
            self.img_list = f.readlines()
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.resize_transform = Resize((224, 224))
        self.image_format = image_format

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx].strip())
        image = read_image(img_path)
        if self.image_format == "RGB":
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
        elif self.image_format == "grayscale":
            if image.shape[0] == 3:
                image = image.mean(dim=0, keepdim=True)
        image = self.resize_transform(image)   
        if self.transform:
            image = self.transform(image)
        return image


if __name__ == "__main__":
    # intialize the dataset
    Dataset = MiniImageNet(img_list='../dataset/MiniImageNet/image_list.txt', img_dir='../dataset/MiniImageNet/')
    train_dataloader = DataLoader(Dataset, batch_size=32, shuffle=True)

    # Display an image from the batch
    train_features = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    img = train_features[0].permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
    plt.imshow(img)
    plt.show()









