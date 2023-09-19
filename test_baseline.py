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

# define a function for mathcing feature_2D (512) to Matrix with size (512 * 1000) colomn-wise with cosine similarity
def clip_similarity(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return text_probs


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
    # Step 1: Load CLIP model
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    clip_model.to(device)
    clip_model.eval()

    # Step 2: Load Realistic Projection object
    proj = Realistic_Projection()

    

    #load the text features
    prompts = read_txt_file("class_name.txt")
    text_features_tmp = torch.zeros((len(prompts), 512), device=device)
    text = 0
    for prompt in prompts:
        text = open_clip.tokenize(["A depth map of" + prompt])
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features_tmp[prompts.index(prompt)] = clip_model.encode_text(text)
        
    text_features = torch.zeros((int(len(prompts)), 512), device=device)
    text_features = text_features_tmp
 

    


    # # load the text features
    # prompts = read_txt_file("chatgpt_description.txt")
    # text_features_tmp = torch.zeros((len(prompts), 512), device=device)
    # text = 0
    # for prompt in prompts:
    #     text = open_clip.tokenize([prompt])
    #     with torch.no_grad(), torch.cuda.amp.autocast():
    #         text_features_tmp[prompts.index(prompt)] = clip_model.encode_text(text)
        
    # text_features = torch.zeros((int(len(prompts)/4), 512), device=device)
    # for i in range(37):
    #     text_features[i,:] = torch.mean(text_features_tmp[4*i:4*i+4,:], dim=0)
    #  #   text_features[i,:] = text_features_tmp[(4*i),:]


    # score prediction
    novel_class_correct = 0
    novel_class_total = 475
    base_class_correct = 0
    base_class_total = 1496

    for j, data in tqdm(enumerate(testloader, 0)):
        points, target = data['pointclouds'].to(device).float(), data['labels'].to(device)
        points, target = points.to(device), target.to(device)

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

   
        features = features_2D
        pred = clip_similarity(features_2D, text_features)
        pred_choice = pred.data.max(1)[1]
        if pred_choice == target:
            if target < 26:
                base_class_correct = base_class_correct + 1
            else:
                novel_class_correct = novel_class_correct + 1


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
