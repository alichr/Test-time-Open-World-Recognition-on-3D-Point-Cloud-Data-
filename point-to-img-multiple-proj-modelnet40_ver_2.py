import argparse
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import open_clip
from utils.mv_utils_zs_ver_2 import Realistic_Projection_Learnable_new as Realistic_Projection 
from model.PointNet import PointNetfeat, feature_transform_regularizer
from model.Transformation import Transformation
from utils.dataloader_ModelNet40 import *
import os
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from utils.Loss import ClipLoss, CombinedConstraintLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define a function for mathcing feature_2D (512) to Matrix with size (512 * 1000) colomn-wise with cosine similarity
def clip_similarity(image_features, text_features):
    image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True) 
    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True) 
    text_probs = (100 * image_features_norm @ text_features_norm.T).softmax(dim=-1)
    return text_probs


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# read a txt file line by line and save it in a list, and remove the empty lines
def read_txt_file(file):
    with open(file, 'r') as f:
        array = f.readlines()
    array = ["An image of a " + x.strip() for x in array]
    array = list(filter(None, array))
    return array

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

# define the main function
def main(opt):

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

    # Step 3: Load PointNet model
    model = PointNetfeat(feature_transform=opt.feature_transform)
    model.to(device)

    # Step 4: Load the Transformation model
    transform = Transformation(num_rotations=10)
    transform.to(device)
    
    #load the text features
    prompts = read_txt_file("class_name_modelnet40.txt")
    text = open_clip.tokenize(prompts)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_embedding_all_classes = clip_model.encode_text(text.to(device))

    # define the optimizer
    optimizer = optim.Adam(list(model.parameters()) + list(transform.parameters()), lr=0.001, betas=(0.9, 0.999))

    # load loss function
    clip_loss = ClipLoss(device)
    constraint_loss = CombinedConstraintLoss(num_rotations=10)

    # train the model
    print("=> Start training the model")
    for epoch in range(opt.nepoch):
        # define the loss
        train_loss = 0
        train_correct = 0
        train_total = 0
        model.train()
        transform.train()
        clip_model.train()

        for i, data in tqdm(enumerate(trainDataLoader, 0)):
            points, target = data
            # convert numpy.int32 to torch.int32
            points, target = points.to(device), target.to(device)
            optimizer.zero_grad()
            points = points.transpose(2, 1)
            points_fea,_,_ = model(points)
               
            # Forward samples to the Transformation model
            transform_matrix = transform(points_fea)

            print(transform_matrix.shape)

            loss_orthogonal = constraint_loss(transform_matrix.to(device)).mean()

            print(loss_orthogonal)
            stop
                
            # Project samples to an image surface to generate 3 depth maps
            depth_map = torch.zeros((points.shape[0] * 10, 3, 110, 110)).to(device)
            points = points.transpose(2, 1)   
            for k in range(10):
                    depth_map_tmp = proj.get_img(points, transform_matrix[:, k, :, :].view(-1, 9))
                    depth_map[k * points.shape[0]:(k + 1) * points.shape[0]] = depth_map_tmp    
            depth_map = torch.nn.functional.interpolate(depth_map, size=(224, 224), mode='bilinear', align_corners=True)
           
            # Forward samples to the CLIP model
            pc_img = clip_model.encode_image(depth_map).to(device)
            pc_img_avg = torch.zeros((points.shape[0], 512)).to(device)
            for k in range(16):
                rows_to_average = [0 + k, 16 + k, 32 + k, 48 + k, 64 + k, 80 + k, 96 + k, 112 + k, 128 + k, 144 + k]
                pc_img_avg[k] = torch.mean(pc_img[rows_to_average], dim=0)
            img_embedding = pc_img_avg
            # text embedding
            text_embedding = text_embedding_all_classes[target]

            # Calculating the Loss
            loss = clip_loss(img_embedding, text_embedding)
            print(loss)
            loss.backward()
            optimizer.step()

            # Calculating the accuracy
            train_loss += loss.item()
            logits = clip_similarity(img_embedding, text_embedding_all_classes)
            _, predicted = torch.max(logits.data, 1)
            print(predicted)
            print(target)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            print(train_correct)
            print('-------------------------------------')
            # delete the variables to free the memory
            del points, target, points_fea, transform_matrix, depth_map, pc_img, pc_img_avg, img_embedding, text_embedding, loss, logits, predicted
            torch.cuda.empty_cache()

        print(f"=> Epoch {epoch} loss: {train_loss:.2f} accuracy: {100 * train_correct / train_total:.2f}")
        torch.save(model.state_dict(), '%s/3D_model_%d.pth' % (opt.outf, epoch))
        torch.save(transform.state_dict(), '%s/3D_transform_%d.pth' % (opt.outf, epoch))


        # evaluate the model       
        base_class_correct = 0
        base_class_total = 0
        Logits = torch.zeros(2468,40).to(device)
        Target = torch.zeros(2468).to(device)

        model.eval()
        transform.eval()
        clip_model.eval()

        for j, data in tqdm(enumerate(testDataLoader, 0)):
            points, target = data
            # convert numpy.int32 to torch.int32
            points, target = points.to(device), target.to(device)
            features_2D = torch.zeros((1, 512), device=device)
            with torch.no_grad():
                    
                    depth_map = torch.zeros((points.shape[0] * 10, 3, 110, 110)).to(device)
                    # Forward samples to the PointNet model
                    points = points.transpose(2, 1)

                    points_fea,_,_ = model(points)
                    points_fea = points_fea.repeat(2, 1) 
                
                    # Forward samples to the Transformation model
                    transform_matrix = transform(points_fea)

                    # Project samples to an image surface to generate 3 depth maps
                    points = points.transpose(2, 1)   
                    points = points.repeat(2, 1, 1)     
                    
                    for k in range(10):
                            depth_map_tmp = proj.get_img(points, transform_matrix[:, k, :, :].view(-1, 9))
                            depth_map_tmp = depth_map_tmp[0].unsqueeze(0)
                            depth_map[k] = depth_map_tmp   
                    # imshow all three depth maps    
                    depth_map = torch.nn.functional.interpolate(depth_map, size=(224, 224), mode='bilinear', align_corners=True)

                    # Forward samples to the CLIP model
                    pc_img = clip_model.encode_image(depth_map).to(device)
                    # Average the features
                    pc_img_avg = torch.mean(pc_img, dim=0)
                    # Save feature vectors
                    img_features = pc_img_avg.unsqueeze(0)
            
            logits = clip_similarity(img_embedding, text_embedding_all_classes)
            Logits[j] = logits
            Target[j] = target

        acc, _ = accuracy(Logits, Target, topk=(1, 5))
        acc = (acc / Target.shape[0]) * 100
        print(f"=> zero-shot accuracy: {acc:.2f}")
        # put the models in the training mode
        model.train()
        transform.train()
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