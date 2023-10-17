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
from PIL import Image
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
    array = ["An image of " + x.strip() for x in array]
    array = list(filter(None, array))
    return array

def read_txt_file_class_name(file):
    with open(file, 'r') as f:
        array = f.readlines()
    array = [x.strip() for x in array]
    array = list(filter(None, array))
    return array

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

# read json file
def read_json_file(file):
    with open(file, 'r') as f:
        array = json.load(f)
    return array




# define the main function
def main(opt):

    num_rotations = 1
    set_random_seed(opt.manualSeed) 
    # deine data loader
    train_dataset = ModelNetDataLoader(root='dataset/modelnet40_normal_resampled/', args=opt, split='train', process_data=opt.process_data)
    test_dataset = ModelNetDataLoader(root='dataset/modelnet40_normal_resampled/', args=opt, split='test', process_data=opt.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)

    # Step 1: Load CLIP model
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
    clip_model.to(device)
    for param in clip_model.parameters():
        param.requires_grad = False

    # Step 2: Load Realistic Projection object
    proj = Realistic_Projection().to(device)
    # Step 3: Load the Transformation model
    transform = STN3d()
    transform = transform.to(device)
    # Step 4: Load the Relation Network
    relation = RelationNetwork(1024, 512, 256)
    #relation.load_state_dict(torch.load('cls/relation_67.pth',map_location='cpu'))
    relation = relation.to(device)

    # step 5: Load the unet model
    unet = UNetPlusPlus()
    unet = unet.to(device)
    
    
    #load the text features
    
    class_name = read_txt_file_class_name("class_name.txt")
    class_name_prompt = read_txt_file("class_name_modelnet40.txt")
    prompts = read_json_file("modelnet40_1000.json")
   

     # evaluate the model       
    base_class_correct = 0
    base_class_total = 0
    Logits = torch.zeros(2468,40).to(device)
    Target = torch.zeros(2468).to(device)


    transform.eval()
    relation.eval() 
    clip_model.eval()
    unet.eval()

        #load the text features
    prompts = read_txt_file("class_name_modelnet40.txt")
    text = open_clip.tokenize(prompts)
    text_embedding_all_classes = clip_model.encode_text(text.to(device))
    for j, data in tqdm(enumerate(testDataLoader, 0)):
        points, target = data
        # convert numpy.int32 to torch.int32
        points, target = points.to(device), target.to(device)
        features_2D = torch.zeros((1, 512), device=device)
        with torch.no_grad():
                for i in range(opt.nepoch):
                    transform.load_state_dict(torch.load('cls/relation_' + str(i) + '.pth',map_location='cpu'))
                    relation.load_state_dict(torch.load('cls/relation_' + str(i) + '.pth',map_location='cpu'))
                    unet.load_state_dict(torch.load('cls/relation_' + str(i) + '.pth',map_location='cpu'))
                    transform.eval()
                    relation.eval()
                    unet.eval()               
                    depth_map = torch.zeros((points.shape[0] * num_rotations, 3, 110, 110)).to(device)
                    # Forward samples to the PointNet model
                    points = points.transpose(2, 1)
                    points = points.repeat(2, 1, 1)     
                    trans = transform(points)
            
                    points = points.transpose(2, 1)   

                    depth_map = proj.get_img(points, trans.view(-1, 9))    
                    depth_map = torch.nn.functional.interpolate(depth_map, size=(224, 224), mode='bilinear', align_corners=True)

                    depth_map_reverse = 1 - depth_map
                    mask = (depth_map_reverse != 0).float()
                    texture_map = unet(mask)
                    RGB_map = depth_map * texture_map

                    # save the RGB_map as an image
                    img = RGB_map[0,:,:,:].squeeze(0).permute(1,2,0).cpu().numpy()
                    img = (img - img.min()) / (img.max() - img.min())
                    img = (img * 255).astype(np.uint8)
                    img = Image.fromarray(img)

                    img.save('3D-to-2D-proj/vase/test' + str(i) + '.png')

                    # clear the cache
                    torch.cuda.empty_cache()

                stop


                
                    

 
                    
                        
                   
              
 

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
