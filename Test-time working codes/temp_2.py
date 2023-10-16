import argparse
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import open_clip
from utils.mv_utils_zs_ver_2 import Realistic_Projection_Learnable_new as Realistic_Projection 
from model.PointNet import PointNetfeat, feature_transform_regularizer, STN3d
from model.Transformation import Transformation
from utils.dataloader_ModelNet40 import *
from model.Relation import RelationNetwork
import os
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from utils.Loss import CombinedConstraintLoss
from model.Unet import UNetPlusPlus
import json
from utils.point_splitter import PointCloudSpliter
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Visualize the point clouds
def visualize_point_cloud(point_cloud, title):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, c='b', marker='.')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        plt.show()


import torch
import numpy as np

class PointCloudProcessor:
    def __init__(self):
        pass
    
    @staticmethod
    def random_transform(point_cloud):
        # Apply random rotation
        rotation_matrix = torch.randn(point_cloud.shape[0], 3, 3)
        rotation_matrix, _ = torch.linalg.qr(rotation_matrix)
        rotated_points = torch.matmul(point_cloud, rotation_matrix)
        
        # Apply random split
        point_cloud_split = torch.zeros(rotated_points.shape[0], rotated_points.shape[1], rotated_points.shape[2])
        original_size = (rotated_points.shape[1], rotated_points.shape[2])
        for i in range(rotated_points.shape[0]):
            split_axis = np.random.randint(0, 3)
            part1 = rotated_points[i][rotated_points[i][:, split_axis] > 0]
            part2 = rotated_points[i][rotated_points[i][:, split_axis] <= 0]
            # make all points in part2 same as one point in part1
            part2 = torch.zeros_like(part2)
            point_cloud_split[i] = torch.cat((part1, part2), 0)
            # use interpolation to resize the point cloud to the original size
        
        return point_cloud_split

# Example usage
processor = PointCloudProcessor()







def random_split(point_cloud):
    point_cloud_split = torch.zeros(point_cloud.shape[0], point_cloud.shape[1], point_cloud.shape[2])
    original_size = (point_cloud.shape[1], point_cloud.shape[2])
    for i in range(point_cloud.shape[0]):
        split_axis = np.random.randint(0, 3)
        part1 = point_cloud[i][point_cloud[i][:, split_axis] > 0]
        part2 = point_cloud[i][point_cloud[i][:, split_axis] <= 0]
        # make all points in part2 same as one point in part1
        part2 = torch.zeros_like(part2)
        point_cloud_split[i] = torch.cat((part1, part2), 0)

        # use interpolation to resize the point cloud to the original size
        
        
    return point_cloud_split
    




          


# define the main function
def main(opt):

    num_rotations = 1
    set_random_seed(opt.manualSeed) 
    # deine data loader
    train_dataset = ModelNetDataLoader(root='dataset/modelnet40_normal_resampled/', args=opt, split='train', process_data=opt.process_data)
    test_dataset = ModelNetDataLoader(root='dataset/modelnet40_normal_resampled/', args=opt, split='test', process_data=opt.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)

    
    for i, data in tqdm(enumerate(trainDataLoader, 0)):
        points, target = data
        points, target = points.to(device), target.to(device)

        # rotate the point cloud in all three axis, randomly
        rotated_points_split = processor.random_transform(points)
       # rotated_points = random_rotation(points)

        # split the point cloud into two parts, based on one of the axis, randomly
       # rotated_points_split = random_split(rotated_points)
        print(rotated_points_split.shape)
        visualize_point_cloud(points[12], 'Original Point Cloud')
       # visualize_point_cloud(rotated_points[12], 'Rotated Point Cloud')
        visualize_point_cloud(rotated_points_split[12], 'Split Point Cloud')
        



       
        

        

      
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default= 32, help='input batch size')
    parser.add_argument('--num_points', type=int, default=2048, help='number of points in each input point cloud')
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
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    opt = parser.parse_args()
    main(opt)
    print("Done!")

