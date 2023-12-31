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






def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    return model, preprocess

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# define a projection module point cloud to image with 5 hidden layers (batch, 1024) to (batch, 3, 224, 224)

    


class Transformation(torch.nn.Module):
    def __init__(self, num_rotations=3):
        super(Transformation, self).__init__()
        self.num_rotations = num_rotations
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 9 * num_rotations)  # Output shape is (batch, 9 * num_rotations)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(9 * num_rotations)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.tanh(x)
        x = x.view(-1, self.num_rotations, 3, 3)
        return x


# Define the combined constraint loss
class CombinedConstraintLoss(nn.Module):
    def __init__(self, ortho_weight=1.0, scaling_weight=1.0, det_weight=1.0, num_rotations=3):
        super(CombinedConstraintLoss, self).__init__()
        self.ortho_weight = ortho_weight
        self.scaling_weight = scaling_weight
        self.det_weight = det_weight
        self.num_rotations = num_rotations

    def forward(self, Rs):
        device = Rs.device

        total_loss = 0.0

        for i in range(self.num_rotations):
            R = Rs[:, i, :, :]  # Select the i-th rotation matrix
            result = torch.matmul(R.transpose(1, 2), R)
            ortho_loss = torch.norm(result - torch.eye(3, device=device), p='fro')  # Frobenius norm

            scaling_factors = torch.diagonal(R, dim1=1, dim2=2)
            scaling_loss = torch.norm(scaling_factors - 1.0, p='fro')  # Frobenius norm

            det_loss = torch.abs(torch.det(R) - 1.0)

            total_loss += self.ortho_weight * ortho_loss + self.scaling_weight * scaling_loss + self.det_weight * det_loss

        total_loss /= self.num_rotations

        # Add orthogonal constraint among rotation matrices
        for i in range(self.num_rotations):
            for j in range(i+1, self.num_rotations):
                result = torch.matmul(Rs[:, i, :, :].transpose(1, 2), Rs[:, j, :, :])
                ortho_loss = torch.norm(result, p='fro')  # Frobenius norm
                total_loss += self.ortho_weight * ortho_loss

        return total_loss




constraint_loss = CombinedConstraintLoss()


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



# define a function to convert a target to a class name with this prompt: A image of a [class name]
def target_to_class_name(target):
    class_name = read_txt_file('class_name.txt')
    prompts = []
    calss_names = []
    for i in range(len(target.cpu().numpy())):
        prompt = "A depth map of " + class_name[int(target[i].cpu().numpy())]
        prompts.append(prompt)
        calss_names.append(class_name[int(target[i].cpu().numpy())])
    return prompts, calss_names





def main(opt):
    # Check if a CUDA-enabled GPU is available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set random seed for reproducibility
    set_random_seed(opt.manualSeed)


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

    #for param in feat_ext_3D.parameters():
       # param.requires_grad = False

    for param in clip_model.parameters():
        param.requires_grad = False


    # Create Realistic Projection object
    proj = Realistic_Projection_Learnable_new()
    # define projection module point cloud to image
    Trans = Transformation().to(device)

    # optimizer
    optimizer = optim.Adam(list(Trans.parameters()) + list(feat_ext_3D.parameters()), lr=0.001, betas=(0.9, 0.999))  # Adjust learning rate if needed
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    num_batch = len(trainloader)

    # train the model
    Trans.train()

    kk = 0
    k = 0
    kkk = 0
    jj = 0
    Loss = 0

    for epoch in range(opt.nepoch):
        
        for i, data in enumerate(trainloader):
            if jj == 0:
               sample_identifier_to_track = data['pcd_path'][0]
               print(sample_identifier_to_track)
               jj = 1
            points, target = data['pointclouds'].to(device).float(), data['labels'].to(device)
            points, target = points.to(device), target.to(device)
            if points.shape[0] == 32:

                optimizer.zero_grad()
                points = points.transpose(2, 1)
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
                
                # extract img feature from clip
                image_embeddings_tmp = clip_model.encode_image(depth_map).to(device)
                image_embeddings = torch.zeros((points.shape[0], 512)).to(device)
                for k in range(32):
                    rows_to_average = [0 + k, 32 + k, 64 + k]
                    image_embeddings[k] = torch.mean(image_embeddings_tmp[rows_to_average, :], dim=0)
                
                # extract features from text clip
                prompts, class_names = target_to_class_name(target)
                prompts_token = open_clip.tokenize(prompts).to(device)
                text_embeddings = clip_model.encode_text(prompts_token).to(device)

                # save img

                if sample_identifier_to_track in data['pcd_path']:
                    sample_index_in_batch = data['pcd_path'].index(sample_identifier_to_track)
                    tracked_sample = depth_map[sample_index_in_batch]
                    for s in range(3):
                        img = depth_map[sample_index_in_batch + (32*s),:,:,:].squeeze(0).permute(1,2,0).detach().cpu().numpy()
                        img = (img - img.min()) / (img.max() - img.min())
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
                print(LOSS)
    

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
    

