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
from utils.datautil_3D_memory_incremental_modelnet_to_scanobjectnn import *
from model.Relation import RelationNetwork
import os
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from utils.Loss import CombinedConstraintLoss
from model.Unet import UNetPlusPlus
from torchmetrics.functional.image import image_gradients
from configs.modelnet_scanobjectNN_info import task_ids_total as tid
import json
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    set_random_seed(opt.manualSeed) 
    
    # import pointnet model
    pointnet = PointNetfeat(global_feat=True, feature_transform=opt.feature_transform)
    pointnet = pointnet.to(device)
    pointnet.load_state_dict(torch.load('cls/pointnet_5.pth', map_location=device))

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
        transform[str(i)].load_state_dict(torch.load('cls/transform_5_%d.pth' % i, map_location=device))

    # load the Unet model
    unet = UNetPlusPlus().to(device)
    unet.load_state_dict(torch.load('cls/unet_5.pth', map_location=device))
   
    # Step 4: Load the Relation Network
    relation = RelationNetwork(1024, 512, 256)
    relation = relation.to(device)
    relation.load_state_dict(torch.load('cls/relation_5.pth', map_location=device))

    
    #load the text features
    class_name = read_txt_file_class_name("class_name.txt")
    class_name_prompt = read_txt_file("class_name_modelnet40.txt")
    prompts = read_json_file("modelnet40_1000.json")
    
    optimizer = optim.Adam(pointnet.parameters(), lr=0.001, betas=(0.9, 0.999))

    # load loss function
    cross_entrpy = nn.BCELoss()
    constraint_loss = CombinedConstraintLoss(num_rotations=num_rotations)
    loss_orthogonal_weight = 0.01
    mse_loss = nn.MSELoss()

    # constract a memory bank of inpt data consisting of 1 samples per calss
    memory_bank = torch.zeros((37, 1024,3)).to(device)
    memory_bank_label = torch.zeros((37, 1)).to(device)
   
    # load the data
    prototype = np.zeros((37, 512))
    sample_num = np.zeros((37))
    for t in range(0,5):
        path=Path(opt.dataset_path)
        print(path)
        dataloader = DatasetGen(opt, root=path, fewshot=5)
        dataset = dataloader.get(t,'training')
        trainDataLoader = dataset[t]['train']
        testDataLoader = dataset[t]['test'] 
        if t == 0:
            num_category = 26
        elif t == 1:
            num_category = 30
        elif t == 2:
            num_category = 34
        else:
            num_category = 37     
        print('task:', t)
        # train the model
        clip_model.eval()
        for i in range(num_rotations):
            transform[format(i)].eval()
        unet.train()
        relation.eval()
        pointnet.eval()
        print("=> Start training the model")
        # construct the memory bank
        for epoch in range(1):
            # define the loss
            for i, data in tqdm(enumerate(trainDataLoader, 0)):
                points, target = data['pointclouds'].to(device).float(), data['labels'].to(device)
                points, target = points.to(device), target.to(device)

                optimizer.zero_grad()
                points = points.transpose(2, 1)

                # Forward samples to the PointNet model
                points_embedding ,_ ,_ = pointnet(points)

                # transformation module
                trans = torch.zeros((points.shape[0], num_rotations, 3, 3), device=device)
                for jj in range(num_rotations):
                    trans[:, jj, :, :] = transform[format(jj)](points)
                            
                # depth map generation
                points = points.transpose(2, 1)   
                depth_map = torch.zeros((points.shape[0] * num_rotations, 3, 224, 224)).to(device)  
                for jj in range(num_rotations):
                    depth_map_tmp = proj.get_img(points, trans[:,jj,:,:].view(-1, 9))    
                    depth_map_tmp = torch.nn.functional.interpolate(depth_map_tmp, size=(224, 224), mode='bilinear', align_corners=True)
                    depth_map[jj * points.shape[0]:(jj + 1) * points.shape[0], :, :, :] = depth_map_tmp
                
                loss_gradient = 0
                RGB_map = torch.zeros((points.shape[0] * num_rotations, 3, 224, 224)).to(device)
                for jj in range(num_rotations):
                    # unet model
                    depth_map_reverse = 1 - depth_map[jj * points.shape[0]:(jj + 1) * points.shape[0]]
                    mask = (depth_map_reverse != 0).float()
                    texture_map = unet(mask)
                    # loss for gradient
                    RGB_map[jj * points.shape[0]:(jj + 1) * points.shape[0], :, :, :] = depth_map[jj * points.shape[0]:(jj + 1) * points.shape[0]] * texture_map

                # Forward samples to the vision CLIP model
                img_embedding_tmp = clip_model.encode_image(RGB_map).to(device)
                img_embedding = 0
                for jj in range(num_rotations):
                    img_embedding += img_embedding_tmp[jj * points.shape[0]:(jj + 1) * points.shape[0], :]/ num_rotations
                
                # merge img_embedding and points_embedding
                img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
                points_embedding = points_embedding / points_embedding.norm(dim=-1, keepdim=True)
                
                fea_embedding = (img_embedding + points_embedding)/2

                # calculate the prototype of each class
               
                for jj in range(num_category):
                    prototype[jj, :] += (fea_embedding[target == jj, :].sum(dim=0)).detach().cpu().numpy()
                    sample_num[jj] += ((target == jj).sum()).detach().cpu().numpy()
                print("sample_num:", sample_num)
    for k in range(num_category):
        prototype[k, :] = prototype[k, :] / sample_num[k]
            

    Distance = np.zeros((37))
    print('----------------------------------------------------------')
    for t in range(0,5):
        path=Path(opt.dataset_path)
        print(path)
        dataloader = DatasetGen(opt, root=path, fewshot=5)
        dataset = dataloader.get(t,'training')
        trainDataLoader = dataset[t]['train']
        testDataLoader = dataset[t]['test'] 
        if t == 0:
            num_category = 26
        elif t == 1:
            num_category = 30
        elif t == 2:
            num_category = 34
        else:
            num_category = 37     
        print('task:', t)
        # train the model
        clip_model.eval()
        for i in range(num_rotations):
            transform[format(i)].eval()
        unet.train()
        relation.eval()
        pointnet.eval()
        print("=> Start training the model")
        # construct the memory bank
        for epoch in range(1):
            # define the loss
            for i, data in tqdm(enumerate(trainDataLoader, 0)):
                points, target = data['pointclouds'].to(device).float(), data['labels'].to(device)
                points, target = points.to(device), target.to(device)


                optimizer.zero_grad()
                points = points.transpose(2, 1)

                # Forward samples to the PointNet model
                points_embedding,_,_ = pointnet(points)

                # transformation module
                trans = torch.zeros((points.shape[0], num_rotations, 3, 3), device=device)
                for jj in range(num_rotations):
                    trans[:, jj, :, :] = transform[format(jj)](points)
                            
                # depth map generation
                points = points.transpose(2, 1)   
                depth_map = torch.zeros((points.shape[0] * num_rotations, 3, 224, 224)).to(device)  
                for jj in range(num_rotations):
                    depth_map_tmp = proj.get_img(points, trans[:,jj,:,:].view(-1, 9))    
                    depth_map_tmp = torch.nn.functional.interpolate(depth_map_tmp, size=(224, 224), mode='bilinear', align_corners=True)
                    depth_map[jj * points.shape[0]:(jj + 1) * points.shape[0], :, :, :] = depth_map_tmp
                
                loss_gradient = 0
                RGB_map = torch.zeros((points.shape[0] * num_rotations, 3, 224, 224)).to(device)
                for jj in range(num_rotations):
                    # unet model
                    depth_map_reverse = 1 - depth_map[jj * points.shape[0]:(jj + 1) * points.shape[0]]
                    mask = (depth_map_reverse != 0).float()
                    texture_map = unet(mask)
                    # loss for gradient
                    RGB_map[jj * points.shape[0]:(jj + 1) * points.shape[0], :, :, :] = depth_map[jj * points.shape[0]:(jj + 1) * points.shape[0]] * texture_map

                # Forward samples to the vision CLIP model
                img_embedding_tmp = clip_model.encode_image(RGB_map).to(device)
                img_embedding = 0
                for jj in range(num_rotations):
                    img_embedding += img_embedding_tmp[jj * points.shape[0]:(jj + 1) * points.shape[0], :]/ num_rotations
                
                # merge img_embedding and points_embedding
                img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
                points_embedding = points_embedding / points_embedding.norm(dim=-1, keepdim=True)
                fea_embedding = (img_embedding + points_embedding)/2

                # select one sample per class            
                for jj in range(points.shape[0]):
                    dis = torch.cosine_similarity(fea_embedding[jj, :].unsqueeze(0), torch.from_numpy(prototype[target[jj], :]).to(device).unsqueeze(0))
                    if dis > Distance[target[jj]]:
                        memory_bank[target[jj], :, :] = points[jj, :, :]
                        memory_bank_label[target[jj], :] = target[jj]
                        Distance[target[jj]] = dis
            
        # save the memory bank as numpy array
        np.save('memory/memory_bank_modelnet-to-scanobjectnn.npy', memory_bank.cpu().numpy())
        np.save('memory/memory_bank_label_modelnet-to-scanobjectnn.npy', memory_bank_label.cpu().numpy())
        print("memory bank saved")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default= 32, help='input batch size')
    parser.add_argument('--num_points', type=int, default=2048, help='number of points in each input point cloud')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder to save results')
    parser.add_argument('--model', type=str, default='cls/3D_model_249.pth', help='path to load a pre-trained model')
    parser.add_argument('--feature_transform', action='store_true', help='use feature transform')
    parser.add_argument('--manualSeed', type=int, default = 42, help='random seed')
    parser.add_argument('--dataset_path', type=str, default= 'dataset/FSCIL/modelnet_scanobjectnn/', help="dataset path")
    parser.add_argument('--ntasks', type=str, default= '5', help="number of tasks")
    parser.add_argument('--nclasses', type=str, default= '26', help="number of classes")
    parser.add_argument('--task', type=str, default= '0', help="task number")
    parser.add_argument('--num_samples', type=str, default= '0', help="number of samples per class")
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_category', default=26, type=int, choices=[20, 40],  help='training on ModelNet10/40')
    parser.add_argument('--sem_file', default=None,  help='training on ModelNet10/40')
    parser.add_argument('--use_memory', default=False, help='use_memory')
    parser.add_argument('--herding', default=True, help='herding')
    opt = parser.parse_args()
    main(opt)
    print("Done!")
