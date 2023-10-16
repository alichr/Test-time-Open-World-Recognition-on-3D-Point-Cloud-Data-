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

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

# define the main function
def main(opt):

    num_rotations = 2
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
    # define muliple transformations based on STN3d
    transform = {str(i): STN3d() for i in range(num_rotations)}
    for i in range(num_rotations):
        transform[str(i)].to(device)

    # Step 4: Load the Relation Network
    relation = RelationNetwork(1024, 512, 256)
    relation = relation.to(device)
    
    #load the text features
    prompts = read_txt_file("class_name_modelnet40.txt")
    text = open_clip.tokenize(prompts)
    #with torch.no_grad(), torch.cuda.amp.autocast():
    text_embedding_all_classes = clip_model.encode_text(text.to(device))
    

    # define the optimizer
    Parameters = [p for model in transform.values() for p in model.parameters()]
    optimizer = optim.Adam(list(relation.parameters()) + Parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0.0001)

    # load loss function
    cross_entrpy = nn.BCELoss()
    constraint_loss = CombinedConstraintLoss(num_rotations=num_rotations)
    loss_orthogonal_weight = 0.1

    # train the model
    clip_model.train()
    for i in range(num_rotations):
        transform[format(i)].train()
    relation.train()
    print("=> Start training the model")
    for epoch in range(opt.nepoch):
        # define the loss
        train_loss = 0
        train_correct = 0
        train_total = 0
       
        for i, data in tqdm(enumerate(trainDataLoader, 0)):
            points, target = data
            # convert numpy.int32 to torch.int32
            points, target = points.to(device), target.to(device)
            optimizer.zero_grad()
            points = points.transpose(2, 1)

            # foewad samples to multiple transformation models
            trans = torch.zeros((points.shape[0], num_rotations, 3, 3), device=device)
            for jj in range(num_rotations):
                trans[:, jj, :, :] = transform[format(jj)](points)
            loss_orthogonal = constraint_loss(trans).mean()
           
            # Project samples to an image surface to generate 3 depth maps
            points = points.transpose(2, 1) 
            depth_map = torch.zeros((points.shape[0] * num_rotations, 3, 224, 224)).to(device)  
            for jj in range(num_rotations):
                depth_map_tmp = proj.get_img(points, trans[:,jj,:,:].view(-1, 9))    
                depth_map_tmp = torch.nn.functional.interpolate(depth_map_tmp, size=(224, 224), mode='bilinear', align_corners=True)
                depth_map[jj * points.shape[0]:(jj + 1) * points.shape[0], :, :, :] = depth_map_tmp
            
            # Forward samples to the vision CLIP model
            img_embedding_tmp = clip_model.encode_image(depth_map).to(device)
            img_embedding = 0
            for jj in range(num_rotations):
                img_embedding += img_embedding_tmp[jj * points.shape[0]:(jj + 1) * points.shape[0], :]/ num_rotations
            
            # Forward samples to the text CLIP model
            text_embedding = text_embedding_all_classes.to(device)
            
            # forwarding samples to the Relation module
            text_embedding = text_embedding.unsqueeze(0).repeat(opt.batch_size,1,1).to(device)
            img_embedding = img_embedding.unsqueeze(0).repeat(opt.num_category,1,1)
            img_embedding = torch.transpose(img_embedding,0,1).to(device)
            relation_pairs = torch.cat((text_embedding.float(),img_embedding.float()),2).view(-1,1024)
            relations = relation(relation_pairs.float()).view(-1, opt.num_category).to(device)

            # cllculate the loss
            one_hot_labels = (torch.zeros(opt.batch_size, opt.num_category).to(device).scatter_(1, target.long().view(-1,1), 1))
            loss_t = cross_entrpy(relations, one_hot_labels)
            loss = loss_t + loss_orthogonal * loss_orthogonal_weight
          
            loss.backward(retain_graph=True)
            optimizer.step()

            # calculate the accuracy
            train_loss += loss.clone().detach().item()
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
        Logits = torch.zeros(2468,40).to(device)
        Target = torch.zeros(2468).to(device)

        for i in range(num_rotations):
            transform[format(i)].eval()
        relation.eval() 
        clip_model.eval()

        for j, data in tqdm(enumerate(testDataLoader, 0)):
            points, target = data
            # convert numpy.int32 to torch.int32
            points, target = points.to(device), target.to(device)
            with torch.no_grad():
                    
                    # Forward samples to the PointNet model
                    points = points.transpose(2, 1)
                    points = points.repeat(2, 1, 1)  

                      # foewad samples to multiple transformation models
                    trans = torch.zeros((points.shape[0], num_rotations, 3, 3), device=device)
                    print(trans.shape)
                    for jj in range(num_rotations):
                        trans[:, jj, :, :] = transform[format(jj)](points)

                    points = points.transpose(2, 1)  

                    depth_map = torch.zeros((points.shape[0] * num_rotations, 3, 224, 224)).to(device)  
                    for jj in range(num_rotations):
                        depth_map_tmp = proj.get_img(points, trans[:,jj,:,:].view(-1, 9))    
                        depth_map_tmp = torch.nn.functional.interpolate(depth_map_tmp, size=(224, 224), mode='bilinear', align_corners=True)
                        depth_map[jj * points.shape[0]:(jj + 1) * points.shape[0], :, :, :] = depth_map_tmp 
                   
                    # Forward samples to the CLIP model
                    img_embedding_tmp = clip_model.encode_image(depth_map).to(device)
                    img_embedding = 0
                    for jj in range(num_rotations):
                        img_embedding += img_embedding_tmp[jj * points.shape[0]:(jj + 1) * points.shape[0], :]/ num_rotations
                    print(img_embedding.shape)
                    img_embedding = img_embedding[0,:].unsqueeze(0)

                    # Forward samples to the text CLIP model
                    text_embedding = text_embedding_all_classes.to(device)
                    
                    text_embedding = text_embedding.unsqueeze(0).repeat(1,1,1).to(device)
                    

                    img_embedding = img_embedding.unsqueeze(0).repeat(opt.num_category,1,1).to(device)
                    img_embedding = torch.transpose(img_embedding,0,1).to(device)
                    relation_pairs = torch.cat((text_embedding.float(),img_embedding.float()),2).view(-1,1024)
                    relations = relation(relation_pairs.float()).view(-1, opt.num_category).to(device)
                   
                     
            prediction = relations.cpu().detach().numpy()
            prediction = np.argmax(prediction, axis=1)
            if prediction == target.cpu().detach().numpy():
               base_class_correct += 1
            
            #target = target.cpu().detach().numpy()
           # base_class_total += target.shape[0]
            #base_class_correct += np.sum(prediction == target)
            logits = relations.cpu().detach()

            Logits[j] = logits
            Target[j] = target

        acc = (base_class_correct / 2468) * 100
        print(f"=> zero-shot accuracy: {acc:.2f}")
        print('-------------------------------------------------------------------------')
        # put the models in the training mode

        for i in range(num_rotations):
            transform[format(i)].train()
        relation.train()
        clip_model.train()
 

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
