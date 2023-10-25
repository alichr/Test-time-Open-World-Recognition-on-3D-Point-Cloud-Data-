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
    fea_weight = 0.8
    set_random_seed(opt.manualSeed) 
    
    # import pointnet model
    pointnet = PointNetfeat(global_feat=True, feature_transform=opt.feature_transform)
    pointnet = pointnet.to(device)
    pointnet.load_state_dict(torch.load('cls/pointnet_220.pth', map_location=device))

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
        transform[str(i)].load_state_dict(torch.load('cls/transform_220_%d.pth' % i, map_location=device))

    # load the Unet model
    unet = UNetPlusPlus().to(device)
    unet.load_state_dict(torch.load('cls/unet_220.pth', map_location=device))
   
    # Step 4: Load the Relation Network
    relation = RelationNetwork(1536, 2048, 1024)
    relation = relation.to(device)
    relation.load_state_dict(torch.load('cls/relation_220.pth', map_location=device))

    
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

    # # constract a memory bank of inpt data consisting of 1 samples per calss
    # memory_bank = torch.zeros((40, 1024,3)).to(device)
    # memory_bank_label = torch.zeros((40, 1)).to(device)
    # c = 0
    # for t in range(0,5):
    #     path=Path(opt.dataset_path)
    #     print(path)
    #     dataloader = DatasetGen(opt, root=path, fewshot=5)
    #     dataset = dataloader.get(t,'training')
    #     trainDataLoader = dataset[t]['train']
    #     num_class = 20 + t * 5

    #      # Loop over the data
    #     for i, data in tqdm(enumerate(trainDataLoader, 0)):
    #         mm = 0
    #     # Get the input and target
    #         points, target = data['pointclouds'].to(device).float(), data['labels'].to(device)
    #         points, target = points.to(device), target.to(device)
        
    #         # Skip batches that are too small
    #         if points.shape[0] < opt.batch_size:
    #             continue
            
    #          # Update the memory bank with active selection
    #         for c in range(num_class):
    #             class_indices = (target == c).nonzero().view(-1)
    #             if class_indices.numel() > 0:
    #                 sample_index = class_indices[0]
    #                 memory_bank[c,:,:] = points[sample_index,:,:]
    #                 memory_bank_label[c,:] = target[sample_index]
    #                 c += 1
    #                 if c == num_class:
    #                     mm = 1
    #                     break
    #         if mm == 1:
    #             break
    # print('memory bank is constructed')
                    


    # load memory bank as a numpy array
    memory_bank = np.load('memory_bank.npy')
    memory_bank_label = np.load('memory_bank_label.npy')
    memory_bank = torch.from_numpy(memory_bank).to(device)
    memory_bank_label = torch.from_numpy(memory_bank_label).to(device)

 
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
        clip_model.train()
        for i in range(num_rotations):
            transform[format(i)].train()
        unet.train()
        relation.train()
        pointnet.train()
        print("=> Start training the model")
        # construct the memory bank
        memory_bank_task = memory_bank[0:(num_category-5),:,:]
        memory_bank_label_task = memory_bank_label[0:(num_category-5),:]
        mm = 0
        if t == 0:
           nepoch = 0
        else: 
           nepoch = opt.nepoch
           
        for epoch in range(nepoch):
            # define the loss
            train_loss = 0
            train_correct = 0
            train_total = 0
            for i, data in tqdm(enumerate(trainDataLoader, 0)):
                points, target = data['pointclouds'].to(device).float(), data['labels'].to(device)
                points, target = points.to(device), target.to(device)
                if points.shape[0] < opt.batch_size:
                    continue

                # Select 16 samples from memory_bank_task and memory_bank_label_task
                indices = torch.randperm(memory_bank_task.shape[0])[:opt.batch_size]
                memory_bank_task_samples = memory_bank_task[indices, :, :]
                memory_bank_label_task_samples = memory_bank_label_task[indices, :]
                points = torch.cat((points, memory_bank_task_samples), 0)
                target = torch.cat((target, memory_bank_label_task_samples.squeeze(1)), 0)


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
                
                loss_gradient = 0
                RGB_map = torch.zeros((points.shape[0] * num_rotations, 3, 224, 224)).to(device)
                for jj in range(num_rotations):
                    # unet model
                    depth_map_reverse = 1 - depth_map[jj * points.shape[0]:(jj + 1) * points.shape[0]]
                    mask = (depth_map_reverse != 0).float()
                    texture_map = unet(mask)
                    # loss for gradient
                    dy_init, dx_init = image_gradients(mask)
                    dy, dx = image_gradients(texture_map)
                    loss_gradient += mse_loss(dy, dy_init) + mse_loss(dx, dx_init)
                    RGB_map[jj * points.shape[0]:(jj + 1) * points.shape[0], :, :, :] = depth_map[jj * points.shape[0]:(jj + 1) * points.shape[0]] * texture_map

                # Forward samples to the vision CLIP model
                img_embedding_tmp = clip_model.encode_image(RGB_map).to(device)
                img_embedding = 0
                for jj in range(num_rotations):
                    img_embedding += img_embedding_tmp[jj * points.shape[0]:(jj + 1) * points.shape[0], :]/ num_rotations
                
                # merge img_embedding and points_embedding
                img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
                points_embedding = points_embedding / points_embedding.norm(dim=-1, keepdim=True)
                fea_embedding = torch.cat((img_embedding, points_embedding), 1)

                # Sample prompts from prompts dictionary
                tid_all = []
                for h in range(t+1):                
                    tid_all += tid[h]
                prompts_batch = []
                for j in range(num_category):
                    tmp_1 = (class_name[tid_all[j]])
                    tmp_2 = prompts[tmp_1]
                    random_idx = random.randint(0, len(tmp_2)-1)
                    prompts_batch.append(tmp_2[random_idx])
                # Forward samples to the text CLIP model
                text = open_clip.tokenize(prompts_batch)
                text_embedding = clip_model.encode_text(text.to(device))

                # normalize the text embedding
                text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                            
                # forwarding samples to the Relation module
                text_embedding = text_embedding.unsqueeze(0).repeat((opt.batch_size)*2,1,1).to(device)
                fea_embedding = fea_embedding.unsqueeze(0).repeat(num_category,1,1)
                fea_embedding = torch.transpose(fea_embedding,0,1).to(device)
                relation_pairs = torch.cat((text_embedding.float(),fea_embedding.float()),2).view(-1,1536)
                relations = relation(relation_pairs.float()).view(-1, num_category).to(device)

                # cllculate the loss
                one_hot_labels = (torch.zeros((opt.batch_size)*2, num_category).to(device).scatter_(1, target.long().view(-1,1), 1))
                loss_t = cross_entrpy(relations, one_hot_labels)
                loss = loss_t + loss_orthogonal * loss_orthogonal_weight + loss_gradient
                loss.backward(retain_graph=True)
                optimizer.step()

                   

                # Calculating the accuracy
                train_loss += loss.clone().detach().item()
                prediction = relations.cpu().detach().numpy()
                prediction = np.argmax(prediction, axis=1)
                target = target.cpu().detach().numpy()
                train_total += target.shape[0]
                train_correct += np.sum(prediction == target)

                # delete the variables to free the memory
                del points, target, depth_map, img_embedding, text_embedding, loss
                torch.cuda.empty_cache()
            print('Relation Module','Point embedding + img _embedding:',loss_orthogonal_weight, 'number of view', num_rotations)    
            print(f"=> Epoch {epoch} loss: {train_loss:.2f} accuracy: {100 * train_correct / train_total:.2f}")

        # evaluate the model       
        base_class_correct = 0
        base_class_total = 0
       

        for i in range(num_rotations):
            transform[format(i)].eval()
        relation.eval() 
        unet.eval()
        clip_model.eval()
        pointnet.eval()
        #load the text features
        prompts_test = read_txt_file("class_name_modelnet40.txt")
        text = open_clip.tokenize(prompts_test)
        text_embedding_all_classes = clip_model.encode_text(text.to(device))
        task1, task2, task3, task4, task5, task1_total, task2_total, task3_total, task4_total, task5_total = [0] * 10
        tid_all = []
        for h in range(t+1):                
            tid_all += tid[h]

        for j, data in tqdm(enumerate(testDataLoader, 0)):
            points, target = data['pointclouds'].to(device).float(), data['labels'].to(device)
            points, target = points.to(device), target.to(device)
            features_2D = torch.zeros((1, 512), device=device)
            with torch.no_grad():
                    
                    depth_map = torch.zeros((points.shape[0] * num_rotations, 3, 110, 110)).to(device)
                    # Forward samples to the PointNet model
                    points = points.transpose(2, 1)
                    points = points.repeat(2, 1, 1)   
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
                
                    RGB_map = torch.zeros((points.shape[0] * num_rotations, 3, 224, 224)).to(device) 
                    for jj in range(num_rotations):
                        # unet model
                        depth_map_reverse = 1 - depth_map[jj * points.shape[0]:(jj + 1) * points.shape[0]]
                        mask = (depth_map_reverse != 0).float()
                        texture_map = unet(mask)
                        RGB_map[jj * points.shape[0]:(jj + 1) * points.shape[0], :, :, :] = depth_map[jj * points.shape[0]:(jj + 1) * points.shape[0]] * texture_map

                    # Forward samples to the CLIP model
                    img_embedding_tmp = clip_model.encode_image(RGB_map).to(device)
                    img_embedding = 0
                    for jj in range(num_rotations):
                        img_embedding += img_embedding_tmp[jj * points.shape[0]:(jj + 1) * points.shape[0], :]/ num_rotations

                    # merge img_embedding and points_embedding
                    img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
                    points_embedding = points_embedding / points_embedding.norm(dim=-1, keepdim=True)
                    fea_embedding = torch.cat((img_embedding, points_embedding), 1)
                    fea_embedding = fea_embedding[0,:].unsqueeze(0)

                    # Forward samples to the text CLIP model
                    text_embedding = text_embedding_all_classes[tid_all].to(device)
                    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                    
                    # forwarding samples to the Relation module
                    text_embedding = text_embedding.unsqueeze(0).repeat(1,1,1).to(device)
                    fea_embedding = fea_embedding.unsqueeze(0).repeat(num_category,1,1).to(device)
                    fea_embedding = torch.transpose(fea_embedding,0,1).to(device)
                    relation_pairs = torch.cat((text_embedding.float(),fea_embedding.float()),2).view(-1,1536)
                    relations = relation(relation_pairs.float()).view(-1, num_category).to(device)
                
                    
            prediction = relations.cpu().detach().numpy()
            prediction = np.argmax(prediction, axis=1)
            if prediction == target.cpu().detach().numpy():
                base_class_correct += 1
            if prediction == target.cpu().detach().numpy() and target.cpu().detach().numpy() < 20:
                task1 += 1
            if prediction == target.cpu().detach().numpy() and target.cpu().detach().numpy() >= 20 and target.cpu().detach().numpy() < 25:
                task2 += 1
            if prediction == target.cpu().detach().numpy() and target.cpu().detach().numpy() >= 25 and target.cpu().detach().numpy() < 30:
                task3 += 1
            if prediction == target.cpu().detach().numpy() and target.cpu().detach().numpy() >= 30 and target.cpu().detach().numpy() < 35:
                task4 += 1
            if prediction == target.cpu().detach().numpy() and target.cpu().detach().numpy() >= 35 and target.cpu().detach().numpy() < 40:
                task5 += 1
            # tasks total number samples
            if target.cpu().detach().numpy() < 20:
                task1_total += 1
            if target.cpu().detach().numpy() >= 20 and target.cpu().detach().numpy() < 25:
                task2_total += 1
            if target.cpu().detach().numpy() >= 25 and target.cpu().detach().numpy() < 30:
                task3_total += 1
            if target.cpu().detach().numpy() >= 30 and target.cpu().detach().numpy() < 35:
                task4_total += 1
            if target.cpu().detach().numpy() >= 35 and target.cpu().detach().numpy() < 40:
                task5_total += 1

        acc = (base_class_correct / testDataLoader.__len__()) * 100
        
        if task1_total > 0:
           print('task1:', task1/task1_total)
        if task2_total > 0:
           print('task2:', task2/task2_total)
        if task3_total > 0:
           print('task3:', task3/task3_total)
        if task4_total > 0:
           print('task4:', task4/task4_total)
        if task5_total > 0:
           print('task5:', task5/task5_total)
        print(f"=> total accuracy: {acc:.2f}")
        print('-------------------------------------------------------------------------')
        # put the models in the training mode

        for i in range(num_rotations):
            transform[format(i)].train()
        relation.train()
        unet.train()
        clip_model.train()
        pointnet.train()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default= 16, help='input batch size')
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
