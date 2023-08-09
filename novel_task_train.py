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
import os



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define subspace memory
def subsapce():
    """
    Create and return a subsapce memory.
    """
    Subsapce = {}
    for file in os.listdir("subspace/"): 
        Subsapce[file[:-4]] = np.load("subspace/"+file)
    return Subsapce

# define a function for pointnet
def pointnet():
    """
    Create and return a PointNet model.

    Returns:
    - model: The PointNet model.
    """
    model = PointNetfeat(global_feat=True, feature_transform='True').to(device)
    return model

## Define CLIP model
def clip_model():
    """
    Create and return a CLIP model with a specified architecture and pre-trained weights.

    Returns:
    - model: The CLIP model.
    - preprocess: The preprocessing function to prepare images for the model.
    """
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    return model, preprocess

## Define projection
def projection():
    """
    Create and return a Realistic Projection object.

    Returns:
    - proj: Realistic Projection object.
    """
    proj = Realistic_Projection()
    return proj


## Define subsapce mathcing function using subspace memory with the following formulation: distance = norm(feature - (feature * subspace.T) * subspace)
import torch

def subspace_matching(features, Subspace):
    """
    Match features to a subspace memory.

    Args:
    - features: The features to match.
    - Subspace: The subspace memory.
    - device: The device to use (e.g., 'cuda' or 'cpu').

    Returns:
    - distance: The distance between the features and the subspace memory.
    """
    distance = torch.zeros((features.shape[0], len(Subspace.keys())), device=device)
    for i, key in enumerate(Subspace.keys()):
        subspace_matrix = torch.from_numpy(Subspace[key]).to(device).float()
        distance[:, i] = torch.norm(features - torch.matmul(torch.matmul(features, subspace_matrix), subspace_matrix.transpose(0, 1)), dim=1)
    return distance








# define the main function
def main(opt):
    Distance = torch.zeros(1971)
    # deine data loader
    dataloader = DatasetGen(opt, root=Path(opt.dataset_path), fewshot=argument.fewshot)
    t = 1
    dataset = dataloader.get(t,'Test')
    testloader = dataset[t]['test']
    print("Test dataset size:", len(testloader.dataset))

    # Step 0: Load PointNet model
    feature_ext_3D = pointnet()
    feature_ext_3D = feature_ext_3D.to(device)
    feature_ext_3D.load_state_dict(torch.load(opt.model, map_location=torch.device('cpu')))
    feature_ext_3D.eval()

    # Step 1: Load CLIP model
    feature_ext_2D, preprocess = clip_model()
    feature_ext_2D = feature_ext_2D.to(device)
    # Step 2: Load Realistic Projection object
    proj = projection()

    # Step 3: Load subspace memory
    Subsapce = subsapce()

    # step4: define the classifer
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

    # Load the classifier weights
    classifier.load_state_dict(torch.load('cls/cls_model_249.pth', map_location=torch.device('cpu')))

    total_correct = 0
    total_testset = 0
    for j, data in tqdm(enumerate(testloader, 0)):
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
                pc_img = feature_ext_2D.encode_image(pc_img).to(device)
                # Average the features
                pc_img_avg = torch.mean(pc_img, dim=0)
                # Save feature vectors
                features_2D[i,:] = pc_img_avg

        # Extract 3D features from PointNet
        points = points.transpose(2, 1)
        features_3D, _, _ = feature_ext_3D(points)

        # Concatenate 2D and 3D features
        features = torch.cat((features_2D, features_3D), dim=1)
        
        # print subsapce shape
        print(Subsapce['subspace0'].shape)
        # subsapce matching
        distance = subspace_matching(features, Subsapce)

        Distance[j] = torch.mean(distance)

        print('sample_' + str(j) + ':' , Distance[j])


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
    # convert Distance array to numpy and save as npy file
    Distance = Distance.cpu().detach().numpy()
    np.save('Distance.npy', Distance)

    # plot the Distance array
    plt.plot(Distance)



 
        
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default= 1, help='input batch size')
    parser.add_argument('--num_points', type=int, default=1024, help='number of points in each input point cloud')
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


    opt = parser.parse_args()

    main(opt)
    print("Done!")
