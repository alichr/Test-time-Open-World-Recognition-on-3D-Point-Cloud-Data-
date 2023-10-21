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
    set_random_seed(opt.manualSeed) 
    path=Path(opt.dataset_path)
    print(path)
    dataloader = DatasetGen(opt, root=path, fewshot=5)
    t = 0
    dataset = dataloader.get(t,'training')
    trainDataLoader = dataset[t]['train']
    testDataLoader = dataset[t]['test'] 
    
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

    # load the Unet model
    unet = UNetPlusPlus().to(device)
    # Step 4: Load the Relation Network
    relation = RelationNetwork(1024, 512, 256)
    relation = relation.to(device)
    
    #load the text features
    class_name = read_txt_file_class_name("class_name.txt")
    class_name_prompt = read_txt_file("class_name_modelnet40.txt")
    prompts = read_json_file("modelnet40_1000.json")


    # define the optimizer
    optimizer = optim.Adam(list(transform.parameters()) + list(relation.parameters())  + list(unet.parameters()), lr=0.001, betas=(0.9, 0.999))

    # load loss function
    cross_entrpy = nn.BCELoss()
    constraint_loss = CombinedConstraintLoss(num_rotations=num_rotations)
    loss_orthogonal_weight = 0.01
    mse_loss = nn.MSELoss()

    # train the model
    clip_model.train()
    transform.train()
    unet.train()
    relation.train()
    print("=> Start training the model")
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

            optimizer.zero_grad()
            points = points.transpose(2, 1)
            
            trans = transform(points)
                        
            loss_orthogonal = constraint_loss(trans.unsqueeze(1)).mean()

            # Project samples to an image surface to generate 3 depth maps
            points = points.transpose(2, 1)   
            depth_map = proj.get_img(points, trans.view(-1, 9))    
            depth_map = torch.nn.functional.interpolate(depth_map, size=(224, 224), mode='bilinear', align_corners=True)    
            
            # unet model
            depth_map_reverse = 1 - depth_map
            mask = (depth_map_reverse != 0).float()
            texture_map = unet(mask)
            # loss for gradient
            dy_init, dx_init = image_gradients(mask)
            dy, dx = image_gradients(texture_map)
            loss_gradient = mse_loss(dy, dy_init) + mse_loss(dx, dx_init)

            RGB_map = depth_map * texture_map

            # Forward samples to the vision CLIP model
            img_embedding = clip_model.encode_image(RGB_map).to(device)

            # Sample prompts from prompts dictionary
            prompts_batch = []
            for j in range(opt.num_category):
                tmp_1 = (class_name[tid[t][j]])
                tmp_2 = prompts[tmp_1]
                random_idx = random.randint(0, len(tmp_2)-1)
                prompts_batch.append(tmp_2[random_idx])

   
            # Forward samples to the text CLIP model
            text = open_clip.tokenize(prompts_batch)
            text_embedding = clip_model.encode_text(text.to(device))
            
            # forwarding samples to the Relation module
            text_embedding = text_embedding.unsqueeze(0).repeat(opt.batch_size,1,1).to(device)
            img_embedding = img_embedding.unsqueeze(0).repeat(opt.num_category,1,1)
            img_embedding = torch.transpose(img_embedding,0,1).to(device)
            relation_pairs = torch.cat((text_embedding.float(),img_embedding.float()),2).view(-1,1024)
            relations = relation(relation_pairs.float()).view(-1, opt.num_category).to(device)

            # cllculate the loss
            one_hot_labels = (torch.zeros(opt.batch_size, opt.num_category).to(device).scatter_(1, target.long().view(-1,1), 1))
    
            loss_t = cross_entrpy(relations, one_hot_labels)
            loss = loss_t + loss_orthogonal * loss_orthogonal_weight + loss_gradient

           # print('loss',loss)
            loss.backward(retain_graph=True)
            
            optimizer.step()

            # Calculating the accuracy
            train_loss += loss.clone().detach().item()

            # calculate the accuracy
            prediction = relations.cpu().detach().numpy()
            prediction = np.argmax(prediction, axis=1)
            target = target.cpu().detach().numpy()
            train_total += target.shape[0]
            train_correct += np.sum(prediction == target)

            # delete the variables to free the memory
            del points, target, depth_map, img_embedding, text_embedding, loss
            torch.cuda.empty_cache()
        print('Relation Module','loss_orthogonal_weight:',loss_orthogonal_weight, 'number of view', num_rotations)    
        print(f"=> Epoch {epoch} loss: {train_loss:.2f} accuracy: {100 * train_correct / train_total:.2f}")
        torch.save(transform.state_dict(), '%s/transform_%d.pth' % (opt.outf, epoch))
        torch.save(relation.state_dict(), '%s/relation_%d.pth' % (opt.outf, epoch))


        # evaluate the model       
        base_class_correct = 0
        base_class_total = 0
       

        transform.eval()
        relation.eval() 
        unet.eval()
        clip_model.eval()
        #load the text features
        prompts = read_txt_file("class_name_modelnet40.txt")
        text = open_clip.tokenize(prompts)
        text_embedding_all_classes = clip_model.encode_text(text.to(device))

        for j, data in tqdm(enumerate(testDataLoader, 0)):
            points, target = data['pointclouds'].to(device).float(), data['labels'].to(device)
            # convert numpy.int32 to torch.int32
            points, target = points.to(device), target.to(device)
            print('target',target)
            features_2D = torch.zeros((1, 512), device=device)
            with torch.no_grad():
                    
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
                    
                   
                    # Forward samples to the CLIP model
                    img_embedding = clip_model.encode_image(RGB_map).to(device)
                    img_embedding = img_embedding
                    
                    img_embedding = img_embedding[0,:].unsqueeze(0)

                    # Forward samples to the text CLIP model
                    text_embedding = text_embedding_all_classes[tid[t]].to(device)
                    
                    
                    text_embedding = text_embedding.unsqueeze(0).repeat(1,1,1).to(device)
                    

                    img_embedding = img_embedding.unsqueeze(0).repeat(opt.num_category,1,1).to(device)
                    img_embedding = torch.transpose(img_embedding,0,1).to(device)
                    relation_pairs = torch.cat((text_embedding.float(),img_embedding.float()),2).view(-1,1024)
                    relations = relation(relation_pairs.float()).view(-1, opt.num_category).to(device)
                   
                     
            prediction = relations.cpu().detach().numpy()
            prediction = np.argmax(prediction, axis=1)
            print(prediction)
            print(target.cpu().detach().numpy())
            if prediction == target.cpu().detach().numpy():
               base_class_correct += 1
            
            #target = target.cpu().detach().numpy()
           # base_class_total += target.shape[0]
            #base_class_correct += np.sum(prediction == target)
            logits = relations.cpu().detach()


        acc = (base_class_correct / 1958) * 100
        print(f"=> zero-shot accuracy: {acc:.2f}")
        print('-------------------------------------------------------------------------')
        # put the models in the training mode

        transform.train()
        relation.train()
        unet.train()
        clip_model.train()
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default= 16, help='input batch size')
    parser.add_argument('--num_points', type=int, default=2048, help='number of points in each input point cloud')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
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
