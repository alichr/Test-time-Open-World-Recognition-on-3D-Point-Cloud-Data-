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
from torch import nn



def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    return model, preprocess

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# define a projection module point cloud to image
class Projection(torch.nn.Module):
    def __init__(self):
        super(Projection, self).__init__()
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 3*224*224)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(3*224*224)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.sigmoid(x)
        x = x.view(-1, 3, 224, 224)
        return x
    


# read a txt file line by line and save it in a list, and remove the empty lines
def read_txt_file(file):
    """
    Read a txt file line by line and save it in a list, and remove the empty lines.

    Args:
    - file: The file to read.

    Returns:
    - array: The list of lines in the file.
    """
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

    # Define PointNet feature extractor and projection moduele
    feature_ext_3D = PointNetfeat(global_feat=True, feature_transform=opt.feature_transform).to(device)
    
    # define projection module point cloud to image
    proj = Projection().to(device)



  
    # Define optimizer for PointNet and classifier
    parameters = list(feature_ext_3D.parameters()) + list(proj.parameters())
    parameters = [param.to(device) for param in parameters]
    optimizer = optim.Adam(parameters, lr=0.001, betas=(0.9, 0.999))  # Adjust learning rate if needed
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    num_batch = len(trainloader)

    # train the model
    feature_ext_3D.train()
    proj.train()

    kk = 0
    k = 0

    for epoch in range(opt.nepoch):
        
        for i, data in enumerate(trainloader, 0):
            points, target = data['pointclouds'].to(device).float(), data['labels'].to(device)
            points, target = points.to(device), target.to(device)
            optimizer.zero_grad()

            # Extract features from PointNet
            points = points.transpose(2, 1)
            point_embedding, trans, trans_feat = feature_ext_3D(points)

            # point cloud to image
            point_to_img = proj(point_embedding)

            # save image

   


            # extract img feature from clip
            image_embeddings = clip_model.encode_image(point_to_img).to(device)


            # extract features from text clip

            prompts, class_names = target_to_class_name(target)

            prompts_token = open_clip.tokenize(prompts).to(device)
            text_embeddings = clip_model.encode_text(prompts_token).to(device)

            # save img

            kk += 1
            if kk % 100 == 0:
                img = point_to_img[0,:,:,:].squeeze(0).permute(1,2,0).detach().cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min())
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                img.save('3D-to-2D-proj/tmp/' + class_names[0] + '_' + str(k) + '.png')
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

            print(loss)

        

            # Backpropagate
            loss.backward()

            # Update weights
            optimizer.step()


        
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

    class_name = read_txt_file('class_name.txt')


    opt = parser.parse_args()

    ########### constant


    print(opt)
    main(opt)
    