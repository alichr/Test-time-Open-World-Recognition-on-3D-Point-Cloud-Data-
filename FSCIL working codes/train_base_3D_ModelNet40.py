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

    # import pointnet model
    pointnet = PointNetfeat(global_feat=True, feature_transform=opt.feature_transform)
    pointnet = pointnet.to(device)
    
    # define a classifier with three fully connected layers
    class Classifier(nn.Module):
        def __init__(self, k=2):
            super(Classifier, self).__init__()
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, k)
            self.dropout = nn.Dropout(p=0.3)
            self.bn1 = nn.BatchNorm1d(256)
            self.bn2 = nn.BatchNorm1d(128)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.dropout(self.fc2(x))))
            x = self.fc3(x)
            return F.log_softmax(x, dim=1)
    classifier = Classifier(k=opt.num_category).to(device)
    # define the optimizer
    optimizer = optim.Adam(list(classifier.parameters()) + list(pointnet.parameters()), lr=0.001, betas=(0.9, 0.999))

    # load loss function
    cross_entrpy = nn.CrossEntropyLoss()
   
    pointnet.train()
    classifier.train()
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

            # Forward samples to the PointNet model
            points_embedding,_,_ = pointnet(points)
            y_pred = classifier(points_embedding)

            # Calculating the loss
            loss = cross_entrpy(y_pred, target.long())
            loss.backward(retain_graph=True)
            optimizer.step()

            # Calculating the accuracy
            train_loss += loss.clone().detach().item()
            prediction = y_pred.cpu().detach().numpy()
            prediction = np.argmax(prediction, axis=1)
            target = target.cpu().detach().numpy()
            train_total += target.shape[0]
            train_correct += np.sum(prediction == target)

             
        print(f"=> Epoch {epoch} loss: {train_loss:.2f} accuracy: {100 * train_correct / train_total:.2f}")
        torch.save(pointnet.state_dict(), '%s/pointnet_%d.pth' % (opt.outf, epoch))
        torch.save(classifier.state_dict(), '%s/classifier_%d.pth' % (opt.outf, epoch))

        # evaluate the model       
        base_class_correct = 0
        base_class_total = 0
       
        pointnet.eval()
        classifier.eval()

        for j, data in tqdm(enumerate(testDataLoader, 0)):
            points, target = data['pointclouds'].to(device).float(), data['labels'].to(device)
            points, target = points.to(device), target.to(device)
            features_2D = torch.zeros((1, 512), device=device)
            with torch.no_grad():
                    points = points.transpose(2, 1)
                    points = points.repeat(2, 1, 1)   
                    points_embedding,_,_ = pointnet(points)
                    y_pred = classifier(points_embedding)
       
            prediction = y_pred[0].cpu().detach().numpy()
            prediction = np.argmax(prediction, axis=1)
            if prediction == target.cpu().detach().numpy():
               base_class_correct += 1
            
            logits = y_pred.cpu().detach()


        acc = (base_class_correct / 1958) * 100
        print(f"=> zero-shot accuracy: {acc:.2f}")
        print('-------------------------------------------------------------------------')
        # put the models in the training mode


        pointnet.train()
        classifier.train()
 

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
