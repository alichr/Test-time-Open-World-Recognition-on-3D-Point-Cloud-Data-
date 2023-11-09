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
#from utils.dataloader_ModelNet40 import *
from utils.datautil_3D_memory_incremental_modelnet import *
from model.Relation import RelationNetwork
import os
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from utils.Loss import CombinedConstraintLoss
from model.Unet import UNetPlusPlus
from torchmetrics.functional.image import image_gradients
from configs.modelnet_info import task_ids_total as tid
import json
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# read a txt file line by line and save it in a list, and remove the empty lines
def read_txt_file(file):
    with open(file, 'r') as f:
        array = f.readlines()
    array = ["A depth map of " + x.strip() for x in array]
    array = list(filter(None, array))
    return array


def read_txt_file_class_name(file):
    with open(file, 'r') as f:
        array = f.readlines()
    array = [x.strip() for x in array]
    array = list(filter(None, array))
    return array

# read json file
def read_json_file(file):
    with open(file, 'r') as f:
        array = json.load(f)
    return array

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

# define the main function
def main(opt):
    
    num_rotations = 1
    fea_weight = 0.8
    set_random_seed(opt.manualSeed) 
    
    # import pointnet model
    pointnet = PointNetfeat(global_feat=True, feature_transform=opt.feature_transform)
    pointnet = pointnet.to(device)
    #pointnet.load_state_dict(torch.load('cls/pointnet_220.pth', map_location=device))

    # Step 1: Load CLIP model
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
    clip_model.to(device)
    for param in clip_model.parameters():
        param.requires_grad = False

    # Step 2: Load Realistic Projection object
    proj = Realistic_Projection().to(device)
    # Step 3: Load the Transformation model
    transform = {str(i): STN3d() for i in range(num_rotations)}
    for i in range(num_rotations):
        transform[str(i)].to(device)
       # transform[str(i)].load_state_dict(torch.load('cls/transform_220_%d.pth' % i, map_location=device))

    # load the Unet model
    unet = UNetPlusPlus().to(device)
   # unet.load_state_dict(torch.load('cls/unet_220.pth', map_location=device))
   
    # Step 4: Load the Relation Network
    relation = RelationNetwork(1536, 2048, 1024)
    relation = relation.to(device)
   # relation.load_state_dict(torch.load('cls/relation_220.pth', map_location=device))

    
    #load the text features
    class_name = read_txt_file_class_name("class_name.txt")
    class_name_prompt = read_txt_file("class_name_modelnet40.txt")
    prompts = read_json_file("modelnet40_1000.json")
    

    # define the optimizer
    Parameters = [p for model in transform.values() for p in model.parameters()]
    #optimizer = optim.Adam(Parameters + list(relation.parameters())  + list(unet.parameters()) + list(pointnet.parameters()), lr=0.001, betas=(0.9, 0.999))

    optimizer = optim.Adam(relation.parameters(), lr=0.001, betas=(0.9, 0.999))


    # load loss function
    cross_entrpy = nn.BCELoss()
    constraint_loss = CombinedConstraintLoss(num_rotations=num_rotations)
    loss_orthogonal_weight = 0.01
    mse_loss = nn.MSELoss()


 
    for t in range(0,5):
        path=Path(opt.dataset_path)
        print(path)
        dataloader = DatasetGen(opt, root=path, fewshot=5)
        dataset = dataloader.get(t,'training')
        trainDataLoader = dataset[t]['train']
        testDataLoader = dataset[t]['test'] 
        num_category = 20 + t * 5
        print('task:', t)
        # train the model
        clip_model.eval()
        for i in range(num_rotations):
            transform[format(i)].eval()
        unet.eval()
        relation.eval()
        pointnet.eval()
        print("=> Start visulation the shapes")
     
           
        for epoch in range(opt.nepoch):
            # define the loss
            train_loss = 0
            train_correct = 0
            train_total = 0
            for i, data in tqdm(enumerate(trainDataLoader, 0)):
                points, target = data['pointclouds'].to(device).float(), data['labels'].to(device)
                points, target = points.to(device), target.to(device)
                if points.shape[0] < opt.batch_size:
                    continue

                #visulize one of the 3d point cloud using open3d
                # import open3d as o3d
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(points[21].cpu().numpy())
                # o3d.visualization.draw_geometries([pcd])
                # stop
                
                

                optimizer.zero_grad()
                points = points.transpose(2, 1)

                # Forward samples to the PointNet model
                points_embedding,_,_ = pointnet(points)

                # transformation module
                trans = torch.zeros((points.shape[0], num_rotations, 3, 3), device=device)
                for jj in range(num_rotations):
                    trans[:, jj, :, :] = transform[format(jj)](points)
                loss_orthogonal = constraint_loss(trans).mean()
                            
                # depth map generation
                points = points.transpose(2, 1)   
                depth_map = torch.zeros((points.shape[0] * num_rotations, 3, 224, 224)).to(device)  
                for jj in range(num_rotations):
                    depth_map_tmp = proj.get_img(points, trans[:,jj,:,:].view(-1, 9))    
                    depth_map_tmp = torch.nn.functional.interpolate(depth_map_tmp, size=(224, 224), mode='bilinear', align_corners=True)
                    depth_map[jj * points.shape[0]:(jj + 1) * points.shape[0], :, :, :] = depth_map_tmp
                
                # save the depth map as image png
                import torchvision
               # torchvision.utils.save_image(depth_map[21], 'depth_map.png')
                
                


                loss_gradient = 0
                RGB_map = torch.zeros((points.shape[0] * num_rotations, 3, 224, 224)).to(device)
                for jj in range(num_rotations):
                    # unet model
                    depth_map_reverse = 1 - depth_map[jj * points.shape[0]:(jj + 1) * points.shape[0]]
                    mask = (depth_map_reverse != 0).float()
      
                    torchvision.utils.save_image(mask[5], 'mask_map_1.png')
                    
                    texture_map = unet(mask)
                    # loss for gradient
                    dy_init, dx_init = image_gradients(mask)
                    dy, dx = image_gradients(texture_map)
                    loss_gradient += mse_loss(dy, dy_init) + mse_loss(dx, dx_init)
                    RGB_map[jj * points.shape[0]:(jj + 1) * points.shape[0], :, :, :] = depth_map[jj * points.shape[0]:(jj + 1) * points.shape[0]] * texture_map

                torchvision.utils.save_image(RGB_map[5], 'RGB_map_1.png')
                stop
       
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default= 32, help='input batch size')
    parser.add_argument('--num_points', type=int, default=2048, help='number of points in each input point cloud')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder to save results')
    parser.add_argument('--model', type=str, default='cls/3D_model_249.pth', help='path to load a pre-trained model')
    parser.add_argument('--feature_transform', action='store_true', help='use feature transform')
    parser.add_argument('--manualSeed', type=int, default = 42, help='random seed')
    parser.add_argument('--dataset_path', type=str, default= 'dataset/FSCIL/modelnet/', help="dataset path")
    parser.add_argument('--ntasks', type=str, default= '5', help="number of tasks")
    parser.add_argument('--nclasses', type=str, default= '26', help="number of classes")
    parser.add_argument('--task', type=str, default= '0', help="task number")
    parser.add_argument('--num_samples', type=str, default= '0', help="number of samples per class")
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_category', default=20, type=int, choices=[20, 40],  help='training on ModelNet10/40')
    parser.add_argument('--sem_file', default=None,  help='training on ModelNet10/40')
    parser.add_argument('--use_memory', default=False, help='use_memory')
    parser.add_argument('--herding', default=True, help='herding')
    opt = parser.parse_args()
    main(opt)
    print("Done!")
