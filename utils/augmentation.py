import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import csv 
import glob
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from path import Path
import math
import random
from torchvision import transforms, utils,datasets, models
import sys



class Normalize(object):
    def __call__(self, pointcloud):
       # assert len(pointcloud.shape)==2
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))
        return  norm_pointcloud

class RandRotation_z(object):
    def __call__(self, pointcloud):
        #assert len(pointcloud.shape)==2
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        #assert len(pointcloud.shape)==2
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud

class ToNumPy(object):
    def __call__(self, pointcloud):
        #assert len(pointcloud.shape)==2
        return pointcloud.numpy()

class ToTensor(object):
    def __call__(self, pointcloud):
        #assert len(pointcloud.shape)==2
        return torch.from_numpy(pointcloud)
    

# define a class object thta apply the augmentation to the dataset, Normalize, RandRotation_z, RandomNoise, ToTensor
class Augmentation(object):
    def __init__(self):
        self.ToNumPy = ToNumPy()
        self.normalize = Normalize()
        self.randrotation = RandRotation_z()
        self.randomnoise = RandomNoise()
        self.totensor = ToTensor()

    def augment(self, pointcloud):
        pointcloud = self.ToNumPy(pointcloud)
        pointcloud = self.normalize(pointcloud)
        pointcloud = self.randrotation(pointcloud)
        pointcloud = self.randomnoise(pointcloud)
        pointcloud = self.totensor(pointcloud)
        return pointcloud
