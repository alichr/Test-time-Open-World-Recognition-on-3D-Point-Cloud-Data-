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
from model.Unet import UNetPlusPlusCondition, UNetPlusPlus
from model.Transformation import Transformation
from utils.Loss import CombinedConstraintLoss
from torchmetrics.functional.image import image_gradients



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
    for param in clip_model.parameters():
        param.requires_grad = False

    # define pointnet feature extractor
    feat_ext_3D = PointNetfeat(global_feat=True, feature_transform=opt.feature_transform).to(device)
    # define projection module point cloud to image
    Trans = Transformation().to(device)
    # Create Realistic Projection object
    proj = Realistic_Projection_Learnable_new()
    
    # optimizer
    optimizer = optim.Adam(list(Trans.parameters()) + list(feat_ext_3D.parameters()), lr=0.001, betas=(0.9, 0.999))  # Adjust learning rate if needed
    
    # train the model
    Trans.train()
    feat_ext_3D.train()
    kk = 0
    k = 0
    kkk = 0
    jj = 0
    Loss = 0
    kkkk = 0

    for epoch in range(opt.nepoch):
        
        for i, data in enumerate(trainloader):
            if jj == 0:
               sample_identifier_to_track = data['pcd_path'][12]
               print(sample_identifier_to_track)
               jj = 1
            points, target = data['pointclouds'].to(device).float(), data['labels'].to(device)
            points, target = points.to(device), target.to(device)
            if points.shape[0] == 16:

                optimizer.zero_grad()
                points = points.transpose(2, 1)

                # extract elements of Promompt given traget
                prompts, class_names =  target_to_class_name(target)                
                prompts_token = open_clip.tokenize(prompts).to(device)
                text_embeddings = clip_model.encode_text(prompts_token).to(device)

                # extract 3D features
                fea_pointnet, _, _ = feat_ext_3D(points)
                
                # Transformation matrix
                transformation = Trans(fea_pointnet)

                # apply orthogonality to the transformation matrix
                loss_orthogonal = constraint_loss(transformation).mean()
                
                # extract depth map
                points = points.transpose(2, 1)                
                depth_map = torch.zeros((points.shape[0] * 3, 3, 224, 224)).to(device)
                for k in range(3):
                        depth_map_tmp = proj.get_img(points, transformation[:, k, :, :].view(-1, 9))
                        depth_map_tmp = torch.nn.functional.interpolate(depth_map_tmp, size=(224, 224), mode='bilinear', align_corners=True)
                        depth_map[k * points.shape[0]:(k + 1) * points.shape[0]] = depth_map_tmp


                # extract img feature from clip
                image_embeddings_tmp = clip_model.encode_image(depth_map).to(device)
                image_embeddings = torch.zeros((points.shape[0], 512)).to(device)
                for k in range(16):
                    rows_to_average = [0 + k, 16 + k, 32 + k]
                    image_embeddings[k] = torch.mean(image_embeddings_tmp[rows_to_average, :], dim=0)
                
                # save img
                if sample_identifier_to_track in data['pcd_path']:
                    sample_index_in_batch = data['pcd_path'].index(sample_identifier_to_track)
                    tracked_sample = depth_map[sample_index_in_batch]
                    for s in range(3):
                        img = depth_map[sample_index_in_batch + (16*s),:,:,:].squeeze(0).permute(1,2,0).detach().cpu().numpy()
                       # img = (img - img.min()) / (img.max() - img.min())
                        img = (img * 255).astype(np.uint8)
                        img = Image.fromarray(img)
                        img.save('3D-to-2D-proj/tmp/' + class_names[sample_index_in_batch] + '_view_' + str(s) + '_' + str(kkkk) +  '.png')
                    kkkk += 1
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
                Loss += loss
                kkk += 1

                if kkk == 200:
                    print('epoch', epoch, 'total loss', LOSS, 'embedding loss:', loss, 'orthogonal loss:', loss_orthogonal)
                    print('----------------------------------------------------------------------------------------')
                    kkk = 0
                    Loss = 0

                LOSS.backward()
                optimizer.step()

        torch.save(feat_ext_3D.state_dict(), '%s/3D_model_%d.pth' % (opt.outf, epoch))
        torch.save(Trans.state_dict(), '%s/Transformation_%d.pth' % (opt.outf, epoch))

        # test the trained model on the training 
        # a list of 0 to 25
        
        class_label_list = 
        prompts, class_names =  target_to_class_name(target)                
        prompts_token = open_clip.tokenize(prompts).to(device)
        text_embeddings = clip_model.encode_text(prompts_token).to(device)
        feat_ext_3D.eval()
        Trans.eval()
        for j, data in tqdm(enumerate(testloader, 0)):
            points, target = data['pointclouds'].to(device).float(), data['labels'].to(device)
            points, target = points.to(device), target.to(device)
            points = points.transpose(2, 1)
            # extract 3D features
            fea_pointnet, _, _ = feat_ext_3D(points)
                
            # Transformation matrix
            transformation = Trans(fea_pointnet)


        feat_ext_3D.train()
        Trans.train()
        Unet.train()
          



if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
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
    

