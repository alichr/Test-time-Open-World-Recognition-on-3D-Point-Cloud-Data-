# Import necessary libraries and modules
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.optim as optim
import torch.utils.data
from model.PointNet import PointNetfeat, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import open_clip
from utils.mv_utils_zs import Realistic_Projection



################################################## Define parameters ##################################################

# Check if a CUDA-enabled GPU is available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set up command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=1024, help='number of points in each input point cloud')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder to save results')
parser.add_argument('--model', type=str, default='', help='path to load a pre-trained model')
parser.add_argument('--feature_transform', action='store_true', help='use feature transform')

# Parse the command-line arguments
opt = parser.parse_args()
print(opt)


# Set random seed for reproducibility
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


################################################## Define modles ##################################################


# Function to load CLIP model and preprocessing function
def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    return model, preprocess

# Function to create Realistic Projection object
def create_projection():
    proj = Realistic_Projection()
    return proj

# Load CLIP model and preprocessing function
clip_model, clip_preprocess = load_clip_model()
# Create Realistic Projection object
proj = create_projection()
# define PointNet feature extractor
feature_ext_3D = PointNetfeat(global_feat=True, feature_transform=opt.feature_transform).to(device)

# Define the classifier architecture
classifier = torch.nn.Sequential(
    torch.nn.Linear(1536, 512),
    torch.nn.BatchNorm1d(512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.BatchNorm1d(256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 40),
)


################################################## Define optimizer ##################################################
# define optimzer for pointnet and classiifer but not clip
parameters = list(feature_ext_3D.parameters()) + list(classifier.parameters())

# Move the combined parameters to the selected device
parameters = [param.to(device) for param in parameters]

# Define a single optimizer for both models
optimizer = optim.Adam(parameters, lr=0.001, betas=(0.9, 0.999))  # Adjust learning rate if needed
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


################################################## training and evaluation ##################################################




num_batch = len(dataset) / opt.batchSize


     
for epoch in range(opt.nepoch):

    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        # extract 2D features from clip
        features_2D = torch.zeros((points.shape[0], 512))
        with torch.no_grad():
            for i in range(points.shape[0]):
            # Project samples to an image
                pc_prj = proj.get_img(points[i,:,:].unsqueeze(0))
                pc_img = torch.nn.functional.interpolate(pc_prj, size=(224, 224), mode='bilinear', align_corners=True)  
                pc_img = pc_img.to(device)
                # Forward samples to the CLIP model
                pc_img = clip_model.encode_image(pc_img).to(device)
                # Average the features
                pc_img_avg = torch.mean(pc_img, dim=0)   
                # Save feature vectors
                features_2D[i,:] = pc_img_avg
        # extract 3D features from pointnet
        points = points.transpose(2, 1)
        features_3D, trans, trans_feat = feature_ext_3D(points)
        # concatenate 2D and 3D features
        features = torch.cat((features_2D, features_3D), dim=1)
        # classify
        pred = classifier(features)
        # compute loss
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        # backpropagate
        loss.backward()
        # update weights
        optimizer.step()
        scheduler.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))
        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target[:, 0]
            points, target = points.cuda(), target.cuda()
            # define models in eval model to avoid batchnorm and dropout
            feature_ext_3D = feature_ext_3D.eval()
            feature_ext_2D = feature_ext_2D.eval()
            classifier = classifier.eval()
            # extract 2D features from clip
            features_2D = torch.zeros((points.shape[0], 512))
            with torch.no_grad():
                for i in range(points.shape[0]):
                # Project samples to an image
                    pc_prj = proj.get_img(points[i,:,:].unsqueeze(0))
                    pc_img = torch.nn.functional.interpolate(pc_prj, size=(224, 224), mode='bilinear', align_corners=True)  
                    pc_img = pc_img.to(device)
                    # Forward samples to the CLIP model
                    pc_img = clip_model.encode_image(pc_img).to(device)
                    # Average the features
                    pc_img_avg = torch.mean(pc_img, dim=0)   
                    # Save feature vectors
                    features_2D[i,:] = pc_img_avg
            # extract 3D features from pointnet
            points = points.transpose(2, 1)
            features_3D, trans, trans_feat = feature_ext_3D(points)
            # concatenate 2D and 3D features
            features = torch.cat((features_2D, features_3D), dim=1)
            # classify
            pred = classifier(features)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

total_correct = 0
total_testset = 0
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    feature_ext_3D = feature_ext_3D.eval()
    feature_ext_2D = feature_ext_2D.eval()
    classifier = classifier.eval()
    features_2D = torch.zeros((points.shape[0], 512))
    with torch.no_grad():
        for i in range(points.shape[0]):
        # Project samples to an image
            pc_prj = proj.get_img(points[i,:,:].unsqueeze(0))
            pc_img = torch.nn.functional.interpolate(pc_prj, size=(224, 224), mode='bilinear', align_corners=True)  
            pc_img = pc_img.to(device)
            # Forward samples to the CLIP model
            pc_img = clip_model.encode_image(pc_img).to(device)
            # Average the features
            pc_img_avg = torch.mean(pc_img, dim=0)   
            # Save feature vectors
            features_2D[i,:] = pc_img_avg
            # extract 3D features from pointnet
    points = points.transpose(2, 1)
    features_3D, trans, trans_feat = feature_ext_3D(points)
            # concatenate 2D and 3D features
    features = torch.cat((features_2D, features_3D), dim=1)
            # classify
    pred = classifier(features)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))