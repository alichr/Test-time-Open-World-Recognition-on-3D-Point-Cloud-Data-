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



# novel class in the 1000 imagenet classes
novel_class = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
novel_class_dic = {
    26: ["n03337140", "n03742115", "n04550184"],
    27: ["n02791124", "n03376595", "n04099969"],
    28: ["n03179701"],
    29: ["n03782006"],
    30: ["n04239074"],
    31: ["n03961711"],
    32: ["n03201208", "n03982430"],
    33: ["n04344873"],
    34: ["0"],
    35: ["n04344873"],
    36: ["n04447861"]
}

text_data = np.load("Save_Mean_text_feature_image1000.npy",allow_pickle=True)

# define cenert of kmeans memory
def kmeans_cenerts():
    """
    Create and return a kmeans cenerts memory.
    """
    kmeans_cenerts = np.load("kmeans_centroids.npy")
    return kmeans_cenerts



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

# define a function for mathcing feature_2D (512) to Matrix with size (512 * 1000) colomn-wise with cosine similarity
def cosine_similarity(feature_2D, Matrix):
    
    distance = torch.zeros((feature_2D.shape[0], len(Matrix)), device=device)
    for i in range(len(Matrix)):
        distance[:, i] = F.cosine_similarity(feature_2D, Matrix[i]['text_features'], dim=1)
    # max distance index
    index = torch.argmax(distance, dim=1)

    class_pred = (Matrix[index]['Label'])


    return class_pred


# Function to find the key that matches the prediction
def find_matching_key(prediction, category_dict):
    for key, categories in category_dict.items():
        if prediction in categories:
            return key
    return None











## Define subsapce mathcing function using subspace memory with the following formulation: distance = norm(feature - (feature * subspace.T) * subspace)

# def subspace_matching(features, Subspace):
#     """
#     Match features to a subspace memory.

#     Args:
#     - features: The features to match.
#     - Subspace: The subspace memory.
#     - device: The device to use (e.g., 'cuda' or 'cpu').

#     Returns:
#     - distance: The distance between the features and the subspace memory.
#     """
#     distance = torch.zeros((features.shape[0], len(Subspace.keys())), device=device)
#     for i, key in enumerate(Subspace.keys()):
#         subspace_basis = torch.tensor(Subspace[key], dtype=torch.float32, device=device)
        
#         projection = torch.matmul(features, subspace_basis)
#         distance[:, i] = torch.norm(features - torch.matmul(projection, subspace_basis.t()), dim=1)
    
#     return distance

def distance_to_subspace(features, Subspace):
    def projection(P, V):
        return torch.dot(P, V) / torch.dot(V, V) * V

    def distance(P, P_prime):
        return torch.norm(P - P_prime)
    
    distance = torch.zeros((features.shape[0], len(Subspace.keys())), device=device)
    for i, key in enumerate(Subspace.keys()):
        subspace_basis = torch.tensor(Subspace[key], dtype=torch.float32, device=device)
        projection = torch.matmul(features, subspace_basis)
        distance[:, i] = torch.norm(features - torch.matmul(projection, subspace_basis.t()), dim=1)

    return distance

# claculate distance with the center of the kmaeans cluster
def distance_to_center(features,kmeans_cenerts):

    kmeans_cenerts = torch.tensor(kmeans_cenerts, dtype=torch.float32, device=device)
    distance = torch.zeros((features.shape[0], len(kmeans_cenerts)), device=device)
    for i in range(features.shape[0]):
        for j in range(len(kmeans_cenerts)):
            distance[i,j] = torch.norm(features[i,:]-kmeans_cenerts[j,:])
    return distance



# define the main function
def main(opt):
    Distance_base_samples = np.zeros(1496)
    Distance_novel_samples = np.zeros(475)
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
 

    # Step 4: Load kmeans_cenerts
    Kmeans_cenerts = kmeans_cenerts()

    

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
    b = 0
    n = 0
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

        # out-of-distribution detection
        Dis = np.min(distance_to_center(features, Kmeans_cenerts).detach().numpy())
        if Dis > 3.5: # out-of-distribution
            print("out-of-distribution sample")
            predic_class = cosine_similarity(features_2D, text_data)
            pred = find_matching_key(predic_class, novel_class_dic)
        else: # use the classfier for in-distribution
            print("in-distribution sample")
            pred = classifier(features)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct += correct.item()
            total_testset += points.size()[0]

    print("final accuracy:", total_correct / float(total_testset))
    print("total correct:", total_correct)
     # save results and paramerts of model in a log file
    # f = open("log.txt", "a")
    # f.write("final accuracy: %f" % (total_correct / float(total_testset)))
    # f.close()


            

        
        # if target < 26:
        #     # Calculate the distance to the subspace
        #     Distance_base_samples[b] = np.min(distance_to_center(features, Kmeans_cenerts).detach().numpy())
        #     print('base',Distance_base_samples[b],target)
        #     b += 1
        # else:
        #     # Calculate the distance to the subspace
        #     Distance_novel_samples[n] = np.min(distance_to_center(features, Kmeans_cenerts).detach().numpy())
        #     print('novel',Distance_novel_samples[n],target)
        #     n += 1

        
        
    

    # print("final accuracy:", total_correct / float(total_testset))
    #  # save results and paramerts of model in a log file
    # f = open("log.txt", "a")
    # f.write("final accuracy: %f" % (total_correct / float(total_testset)))
    # f.close()
    # # convert Distance array to numpy and save as npy file
   
    # np.save('Distance.npy', Distance)

   # plot Distance_base_samples and Distance_novel_samples in a sample figure with 2 subplots and different colors
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(Distance_base_samples, 'r')
    plt.ylabel('Distance to subspace')
    plt.xlabel('Base samples')
    plt.subplot(2,1,2)
    plt.plot(Distance_novel_samples, 'b')
    plt.ylabel('Distance to subspace')
    plt.xlabel('Novel samples')
    plt.savefig('Distance.png')
    plt.show()



 
        
 


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
