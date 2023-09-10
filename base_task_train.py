
import argparse
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import open_clip
from utils.mv_utils_zs import Realistic_Projection
from model.PointNet import PointNetfeat, feature_transform_regularizer
from utils.dataloader import *


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    return model, preprocess


def create_projection():
    proj = Realistic_Projection()
    return proj


def main(opt):
    # Check if a CUDA-enabled GPU is available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set random seed for reproducibility
    set_random_seed(opt.manualSeed)
    print("Random Seed:", opt.manualSeed)

    # deine data loader
    path = Path(opt.dataset_path)

    dataloader = DatasetGen(opt, root=path, fewshot=argument.fewshot)
    t = 0
    dataset = dataloader.get(t,'training')
    trainloader = dataset[t]['train']

    
    testloader = dataset[t]['test'] 

    # Load CLIP model and preprocessing function
    clip_model, clip_preprocess = load_clip_model()
    clip_model = clip_model.to(device)
    # Create Realistic Projection object
    proj = create_projection()
    # Define PointNet feature extractor
    feature_ext_3D = PointNetfeat(global_feat=True, feature_transform=opt.feature_transform).to(device)

    # Define the classifier architecture
    classifier = torch.nn.Sequential(
        torch.nn.Linear(1536, 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 26),
    ).to(device)

    # Define optimizer for PointNet and classifier
    parameters = list(feature_ext_3D.parameters()) + list(classifier.parameters())
    parameters = [param.to(device) for param in parameters]
    optimizer = optim.Adam(parameters, lr=0.001, betas=(0.9, 0.999))  # Adjust learning rate if needed
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    num_batch = len(trainloader)
    for epoch in range(opt.nepoch):
        
        for i, data in enumerate(trainloader, 0):
            points, target = data['pointclouds'].to(device).float(), data['labels'].to(device)
            points, target = points.to(device), target.to(device)
            optimizer.zero_grad()

            # Extract 2D features from clip
            features_2D = torch.zeros((points.shape[0], 512), device=device)
            with torch.no_grad():
                for j in range(points.shape[0]):
                    # Project samples to an image
                    pc_prj = proj.get_img(points[j,:,:].unsqueeze(0))
                    pc_img = torch.nn.functional.interpolate(pc_prj, size=(224, 224), mode='bilinear', align_corners=True)
                    pc_img = pc_img.to(device)
                    # Forward samples to the CLIP model
                    pc_img = clip_model.encode_image(pc_img).to(device)
                    # Average the features
                    pc_img_avg = torch.mean(pc_img, dim=0)
                    # Save feature vectors
                    features_2D[j,:] = pc_img_avg

            # Extract 3D features from PointNet
            points = points.transpose(2, 1)
            features_3D, trans, trans_feat = feature_ext_3D(points)

        
            
            
            # Concatenate 2D and 3D features
            features = torch.cat((features_2D, features_3D), dim=1)
    

            # Classify
            pred = classifier(features)

            # Compute loss
            loss = F.nll_loss(F.log_softmax(pred, dim=1), target)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001

            
            # Print gradients
            
        

            # Backpropagate
            loss.backward()

            # Update weights
            optimizer.step()
            for name, param in feature_ext_3D.named_parameters(): 
                print(f'Gradient of {name}:')
                print(param.grad)
          #  scheduler.step()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batch_size)))
            if i % 10 == 0:
                k, data = next(enumerate(testloader, 0))
                points, target = data['pointclouds'].to(device).float(), data['labels'].to(device)
                points, target = points.to(device), target.to(device)
                # Set models to eval mode to avoid batchnorm and dropout
                feature_ext_3D.eval()
                classifier.eval()

                # Extract 2D features from clip
                features_2D = torch.zeros((points.shape[0], 512), device=device)
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

                # Extract 3D features from PointNet
                points = points.transpose(2, 1)
                features_3D, _, _ = feature_ext_3D(points)

                

                # Concatenate 2D and 3D features
                features = torch.cat((features_2D, features_3D), dim=1)

                # Classify
                pred = classifier(features)
                loss = F.nll_loss(F.log_softmax(pred, dim=1), target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] test loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item()/float(opt.batch_size)))
               
                
        torch.save(feature_ext_3D.state_dict(), '%s/3D_model_%d.pth' % (opt.outf, epoch))
        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

    total_correct = 0
    total_testset = 0
    for i, data in tqdm(enumerate(testloader, 0)):
        points, target = data['pointclouds'].to(device).float(), data['labels'].to(device)
        points, target = points.to(device), target.to(device)
        feature_ext_3D.eval()
        classifier.eval()

        features_2D = torch.zeros((points.shape[0], 512), device=device)
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

        # Extract 3D features from PointNet
        points = points.transpose(2, 1)
        features_3D, _, _ = feature_ext_3D(points)

        # Concatenate 2D and 3D features
        features = torch.cat((features_2D, features_3D), dim=1)

        # Classify
        pred = classifier(features)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]

    print("final accuracy:", total_correct / float(total_testset))
     # save results and paramerts of model in a log file
    f = open("log.txt", "a")
    f.write("final accuracy: %f" % (total_correct / float(total_testset)))
    f.close()

 


if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_points', type=int, default=1024, help='number of points in each input point cloud')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder to save results')
    parser.add_argument('--model', type=str, default='', help='path to load a pre-trained model')
    parser.add_argument('--feature_transform', default= 'True' , action='store_true', help='use feature transform')
    parser.add_argument('--manualSeed', type=int, default = 42, help='random seed')
    parser.add_argument('--dataset_path', type=str, default= 'dataset/modelnet_scanobjectnn/', help="dataset path")
    parser.add_argument('--ntasks', type=str, default= '1', help="number of tasks")
    parser.add_argument('--nclasses', type=str, default= '26', help="number of classes")
    parser.add_argument('--task', type=str, default= '0', help="task number")
    parser.add_argument('--num_samples', type=str, default= '0', help="number of samples per class")


    opt = parser.parse_args()

    ########### constant


    print(opt)
    main(opt)
