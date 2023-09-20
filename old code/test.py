import argparse
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import open_clip
from utils.mv_utils_zs_ver_2 import Realistic_Projection_Learnable_new as Realistic_Projection
from model.PointNet import PointNetfeat, feature_transform_regularizer
from utils.dataloader import *
import os
from model.Unet import UNetPlusPlus
from model.Transformation import Transformation
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define a function for mathcing feature_2D (512) to Matrix with size (512 * 1000) colomn-wise with cosine similarity
def clip_similarity(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return text_probs



# define a function to convert a target to a class name with this prompt: A image of a [color] [class name]
def extract_info_from_line(line):
    parts = line.strip().split(', ')
    class_name = parts[0]
    color = parts[1]
    rgb_code = [float(x.strip("(')")) for x in parts[2:5]]
    return class_name, color, rgb_code

def get_info_from_file(file_path, line_index):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if 0 <= line_index < len(lines):
            return extract_info_from_line(lines[line_index])
        else:
            return None
        



# read a txt file line by line and save it in a list, and remove the empty lines
def read_txt_file(file):
    """
 The list of lines in the file.
    """
    with open(file, 'r') as f:
        array = f.readlines()
    array = [x.strip() for x in array]
    array = list(filter(None, array))
    return array


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def showimage(img):
   # img = img / 2 + 0    # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# define the main function
def main(opt):
    # deine data loader
    set_random_seed(opt.manualSeed)
    dataloader = DatasetGen(opt, root=Path(opt.dataset_path), fewshot=argument.fewshot)
    t = 1
    dataset = dataloader.get(t,'training')
    trainloader = dataset[t]['train']
    dataset = dataloader.get(t,'Test')
    testloader = dataset[t]['test']
    print("Test dataset size:", len(testloader.dataset))
    # Step 1: Load CLIP model
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    clip_model.to(device)
    for param in clip_model.parameters():
        param.requires_grad = False
    clip_model.eval()
   
    # Step 2: Load Realistic Projection object
    proj = Realistic_Projection().to(device)

    # Step 3: Load PointNet model
    model_3D = PointNetfeat(global_feat=True, feature_transform=True)
    # load the pre-trained model
  #  model_3D.load_state_dict(torch.load('cls/3D_model_89.pth', map_location = torch.device('cpu')))
    model_3D.to(device)
    model_3D.eval()

    # Step 4: Load Transformer model
    Trans = Transformation().to(device)
   # Trans.load_state_dict(torch.load('cls/Transformation_89.pth', map_location = torch.device('cpu')))
    Trans.eval()
    
    # Step 5: Load Unet model
    Unet = UNetPlusPlus().to(device)
 #   Unet.load_state_dict(torch.load('cls/Unet_89.pth', map_location = torch.device('cpu')))
    Unet.eval()
    
    #load the text features
    prompts = read_txt_file("class_name.txt")
    text_features_tmp = torch.zeros((len(prompts), 512), device=device)
    text = 0
    for prompt in prompts:
        text = open_clip.tokenize(["An image of " + prompt])
        with torch.no_grad():
            text_features_tmp[prompts.index(prompt)] = clip_model.encode_text(text)
    text_features = torch.zeros((int(len(prompts)), 512), device=device)
    text_features = text_features_tmp
    


    # score prediction
    novel_class_correct = 0
    novel_class_total = 475
    base_class_correct = 0
    base_class_total = 1496

    for j, data in tqdm(enumerate(testloader, 0)):
        points, target = data['pointclouds'].to(device).float(), data['labels'].to(device)
        points, target = points.to(device), target.to(device)

        with torch.no_grad():
            # Step 1: forward the point cloud to the 3D model
            pc_3D, _, _ = model_3D(points.transpose(1,2))
            # Step 2: forward the point cloud to the Transformer model
            rotation = Trans(pc_3D)
            # Step 3: forward the rotation matrix to the projection model
            depth_map = torch.zeros((points.shape[0] * 3 * 2, 3, 224, 224)).to(device)
            points = points.repeat(2, 1, 1)
            rotation = rotation.repeat(2, 1, 1, 1)
            
            for k in range(3):
                    depth_map_tmp = proj.get_img(points, rotation[:, k, :, :].view(-1, 9))
                    depth_map_tmp = torch.nn.functional.interpolate(depth_map_tmp, size=(224, 224), mode='bilinear', align_corners=True)
                    depth_map[k * points.shape[0]:(k + 1) * points.shape[0]] = depth_map_tmp
            
            # Step 4: forward the depth map to the Unet model and generate RGB image
            depth_map_reverse = 1 - depth_map
            mask = (depth_map_reverse != 0).float()
            
            
            

          #  img_RGB = Unet(depth_map)
           
            # normalize img_RGB between 0 and 1
            img_RGB = depth_map 

            showimage(img_RGB[0])
            



            # Step 5: forward the RGB image to the CLIP model
            pc_img = clip_model.encode_image(img_RGB).to(device)
            
            # Average the features
            features = torch.mean(pc_img, dim=0)

        pred = clip_similarity(features, text_features)
        # index = torch.argmax(pred)

        pred_choice = torch.argmax(pred)
        if pred_choice == target:
            print("correct")
            if target < 26:
                base_class_correct = base_class_correct + 1
            else:
                novel_class_correct = novel_class_correct + 1
        else:

            print("wrong")
            print('pred:', pred)
            print("pred_choice", pred_choice, "target", target)
            print('--------------------------------------')

    # accuarcy for in-distribution
    print("accuarcy for base classes")
    print("final accuracy:", base_class_correct / float(base_class_total))
    print("total correct:", base_class_correct)
    print('--------------------------------------')
    print("accuarcy for novel classes")
    print("final accuracy:", novel_class_correct / float(novel_class_total))
    print("total correct:", novel_class_correct)
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default= 1, help='input batch size')
    parser.add_argument('--num_points', type=int, default=1024, help='number of points in each input point cloud')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder to save results')
    parser.add_argument('--model', type=str, default='cls/3D_model_249.pth', help='path to load a pre-trained model')
    parser.add_argument('--feature_transform', default=False, action='store_true', help='use feature transform')
    parser.add_argument('--manualSeed', type=int, default = 42, help='random seed')
    parser.add_argument('--dataset_path', type=str, default= 'dataset/modelnet_scanobjectnn/', help="dataset path")
    parser.add_argument('--ntasks', type=str, default= '1', help="number of tasks")
    parser.add_argument('--nclasses', type=str, default= '26', help="number of classes")
    parser.add_argument('--task', type=str, default= '0', help="task number")
    parser.add_argument('--num_samples', type=str, default= '0', help="number of samples per class")


    opt = parser.parse_args()

    main(opt)
    print("Done!")
