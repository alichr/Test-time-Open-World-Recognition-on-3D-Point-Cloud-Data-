import argparse
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import open_clip
from utils.mv_utils_zs_ver_2 import Realistic_Projection_Learnable, Realistic_Projection, Realistic_Projection_Learnable_new
from model.PointNet import PointNetfeat, feature_transform_regularizer
from utils.dataloader import *
from utils.dataloader_miniImageNet import *
from PIL import Image
from torch import nn
torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt
from model.Unet import UNetPlusPlus
from model.Transformation import Transformation
from utils.Loss import CombinedConstraintLoss



def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    return model, preprocess

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




# read a txt file line by line and save it in a list, and remove the empty lines
def read_txt_file(file):
    with open(file, 'r') as f:
        array = f.readlines()
    array = [x.strip() for x in array]
    array = list(filter(None, array))
    return array

# convert an array to int
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


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

# extract prompt and RGB code for all classes
Prompts = []
RGB_codes = []
Class_name = []
for i in range(26):
    file_path = 'class_name_color.txt'  # Replace with the actual file path
    line_index = i  # Replace with the desired line index (0-based)
    result = get_info_from_file(file_path, line_index) 
    if result is not None:
        class_name, color, rgb_code = result
        Prompts.append("An image of " + color.lower() + " " + class_name.lower() + '.')
        RGB_codes.append(rgb_code)
        Class_name.append(class_name)


# define a function to convert a target to a class name with this prompt: A image of a [class name]
def target_to_class_name(target):
    class_name = read_txt_file('class_name.txt')
    prompts = []
    calss_names = []
    for i in range(len(target.cpu().numpy())):
        prompt = "An image of " + class_name[int(target[i].cpu().numpy())]
        prompts.append(prompt)
        calss_names.append(class_name[int(target[i].cpu().numpy())])
    return prompts, calss_names



def main(opt):
    # Check if a CUDA-enabled GPU is available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set random seed for reproducibility
    set_random_seed(opt.manualSeed)

    # define the combined constraint loss
    constraint_loss = CombinedConstraintLoss()


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

    # define pointnet feature extractor
    feat_ext_3D = PointNetfeat(global_feat=True, feature_transform=opt.feature_transform).to(device)
    #feat_ext_3D.load_state_dict(torch.load(opt.model))

    # define unet model
    Unet = UNetPlusPlus().to(device)

    #for param in feat_ext_3D.parameters():
       # param.requires_grad = False

    for param in clip_model.parameters():
        param.requires_grad = False


    # Create Realistic Projection object
    proj = Realistic_Projection_Learnable_new()
    # define projection module point cloud to image
    Trans = Transformation().to(device)

    # optimizer
    optimizer = optim.Adam(list(Trans.parameters()) + list(feat_ext_3D.parameters()) + list(Unet.parameters()), lr=0.001, betas=(0.9, 0.999))  # Adjust learning rate if needed
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    num_batch = len(trainloader)

    # train the model
    Trans.train()
    feat_ext_3D.train()
    Unet.train()

    kk = 0
    k = 0
    kkk = 0
    jj = 0
    Loss = 0

    for epoch in range(opt.nepoch):
        
        for i, data in enumerate(trainloader):
            if jj == 0:
               sample_identifier_to_track = data['pcd_path'][5]
               print(sample_identifier_to_track)
               jj = 1
            points, target = data['pointclouds'].to(device).float(), data['labels'].to(device)
            points, target = points.to(device), target.to(device)
            if points.shape[0] == 32:

                optimizer.zero_grad()
                points = points.transpose(2, 1)

                 # extract features from text clip
                prompts = Prompts[int(target)]
                print(prompts)
                stop
                class_name = Class_name[int(target).]
              #  prompts, class_names = target_to_class_name(target)
                prompts_token = open_clip.tokenize(prompts).to(device)
                text_embeddings = clip_model.encode_text(prompts_token).to(device)

                # extract
                fea_pointnet, _, _ = feat_ext_3D(points)
                
                # Transformation matrix
                transformation = Trans(fea_pointnet)

                # apply orthogonality to the transformation matrix
                loss_orthogonal = constraint_loss(transformation).mean()
                
                # extract depth map
                points = points.transpose(2, 1)
                #depth_map = proj.get_img(points, transformation[:, 0, :, :].view(-1, 9))
                
                depth_map = torch.zeros((points.shape[0] * 3, 3, 224, 224)).to(device)
            
                for k in range(3):
                        depth_map_tmp = proj.get_img(points, transformation[:, k, :, :].view(-1, 9))
                        depth_map_tmp = torch.nn.functional.interpolate(depth_map_tmp, size=(224, 224), mode='bilinear', align_corners=True)
                        depth_map[k * points.shape[0]:(k + 1) * points.shape[0]] = depth_map_tmp

                # extact mask from depth map and apply it to the rgb image

                # reverse the depth map and extract the mask
                depth_map_reverse = 1 - depth_map
                mask = (depth_map_reverse != 0).float()
               
                
                # depth map is with size (batch_size * 3, 3, 224, 224), get average in the dimension 1
                depth_map = torch.mean(depth_map, dim=1).unsqueeze(1)
                
                # generate RGB image using unet model
                img_RGB = Unet(depth_map)

                # apply mask to the RGB image
                img_RGB = img_RGB * mask

                # extract img feature from clip
                image_embeddings_tmp = clip_model.encode_image(img_RGB).to(device)
                image_embeddings = torch.zeros((points.shape[0], 512)).to(device)
                for k in range(32):
                    rows_to_average = [0 + k, 32 + k, 64 + k]
                    image_embeddings[k] = torch.mean(image_embeddings_tmp[rows_to_average, :], dim=0)
                
               

                # save img

                if sample_identifier_to_track in data['pcd_path']:
                    sample_index_in_batch = data['pcd_path'].index(sample_identifier_to_track)
                    tracked_sample = img_RGB[sample_index_in_batch]
                    for s in range(3):
                        img = img_RGB[sample_index_in_batch + (32*s),:,:,:].squeeze(0).permute(1,2,0).detach().cpu().numpy()
                       # img = (img - img.min()) / (img.max() - img.min())
                        img = (img * 255).astype(np.uint8)
                        img = Image.fromarray(img)
                        img.save('3D-to-2D-proj/tmp/' + class_names[sample_index_in_batch] + '_' + str(k) + '_view_' + str(s) + '.png')
                    k += 1
                    print('save image')

                # Calculating the Loss
                logits = (text_embeddings @ image_embeddings.T)
                images_similarity = image_embeddings @ image_embeddings.T
                texts_similarity = text_embeddings @ text_embeddings.T
                targets = F.softmax((images_similarity + texts_similarity) / 2 , dim=-1)
                texts_loss = cross_entropy(logits, targets, reduction='none')
                images_loss = cross_entropy(logits.T, targets.T, reduction='none')
                loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
                loss = loss.mean()

                # total loss
                LOSS = loss + (loss_orthogonal / 10)
                print(loss)
    

                Loss += loss
                kkk += 1

                if kkk == 200:
                    print(Loss/200)
                    kkk = 0
                    Loss = 0

                LOSS.backward()
                optimizer.step()

        torch.save(feat_ext_3D.state_dict(), '%s/3D_model_%d.pth' % (opt.outf, epoch))
        torch.save(Trans.state_dict(), '%s/Transformation_%d.pth' % (opt.outf, epoch))

          



if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_points', type=int, default=1024, help='number of points in each input point cloud')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder to save results')
    parser.add_argument('--model', type=str, default='cls/3D_model_249.pth', help='path to load a pre-trained model')
    parser.add_argument('--feature_transform', default= 'False' , action='store_true', help='use feature transform')
    parser.add_argument('--manualSeed', type=int, default = 2, help='random seed')
    parser.add_argument('--dataset_path', type=str, default= 'dataset/modelnet_scanobjectnn/', help="dataset path")
    parser.add_argument('--ntasks', type=str, default= '1', help="number of tasks")
    parser.add_argument('--nclasses', type=str, default= '26', help="number of classes")
    parser.add_argument('--task', type=str, default= '0', help="task number")
    parser.add_argument('--num_samples', type=str, default= '0', help="number of samples per class")

    class_name = read_txt_file('class_name.txt')


    opt = parser.parse_args()

    ########### constant


    print(opt)
    main(opt)
    

