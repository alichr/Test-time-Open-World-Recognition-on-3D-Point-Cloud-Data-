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
from PIL import Image


def create_projection():
    proj = Realistic_Projection()
    return proj


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    return model, preprocess


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

    # Load CLIP model and preprocessing function
    clip_model, clip_preprocess = load_clip_model()
    clip_model = clip_model.to(device)
    # Create Realistic Projection object
    proj = create_projection()
    # Define PointNet feature extractor
   

    num_batch = len(trainloader)
    for epoch in range(opt.nepoch):
        for i, data in enumerate(trainloader, 0):
            points, target = data['pointclouds'].to(device).float(), data['labels'].to(device)
            points, target = points.to(device), target.to(device)
            # Extract 2D features from clip
            features_2D = torch.zeros((points.shape[0], 512), device=device)
            with torch.no_grad():
                for j in range(points.shape[0]):
                    print(target[j])
                    if target[j] == 24:
                        # Project samples to an image
                        pc_prj = proj.get_img(points[j,:,:].unsqueeze(0))
                        pc_img = torch.nn.functional.interpolate(pc_prj, size=(224, 224), mode='bilinear', align_corners=True)
                        # save image
                        for k in range(10):
                            img = pc_img[k,:,:,:].squeeze(0).permute(1,2,0).cpu().numpy()
                            img = (img - img.min()) / (img.max() - img.min())
                            img = (img * 255).astype(np.uint8)
                            img = Image.fromarray(img)
                            # save each image based on k

                            img.save('3D-to-2D-proj/vase/test' + str(k) + '.png')
                        stop
        
                        pc_img = pc_img.to(device)
                        # Forward samples to the CLIP model
                        pc_img = clip_model.encode_image(pc_img).to(device)
                        # Average the features
                        pc_img_avg = torch.mean(pc_img, dim=0)
                        # Save feature vectors
                        features_2D[j,:] = pc_img_avg




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