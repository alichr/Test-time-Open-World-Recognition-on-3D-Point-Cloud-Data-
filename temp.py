import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from utils.dataloader import *
from utils.mv_utils_zs import Realistic_Projection_random
import open_clip


def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    return model, preprocess

class PointCloudToDepthMap(nn.Module):
    def __init__(self, resolution, depth):
        super(PointCloudToDepthMap, self).__init__()
        self.resolution = resolution
        self.depth = depth

    def forward(self, point_cloud, transformation_matrices=None):
        batch_size, num_points, _ = point_cloud.size()

        if transformation_matrices is not None:
            assert transformation_matrices.size(0) == batch_size, "Number of transformation matrices should match batch size"
            assert transformation_matrices.size(1) == 3, "Transformation matrices should have shape (batch_size, 3, 4)"

            # Extend the point_cloud to [batch_size, num_points, 4] with homogeneous coordinate 1
            ones_column = torch.ones((batch_size, num_points, 1), device=point_cloud.device)
            point_cloud_h = torch.cat((point_cloud, ones_column), dim=2)

            # Expand the transformation matrices to (batch_size, 3, 4)
            # and apply to each point in the point cloud
            transformed_points_h = torch.matmul(point_cloud_h, transformation_matrices.permute(0, 2, 1))

            # Remove the homogeneous coordinate
            transformed_points = transformed_points_h[:, :, :3]

        else:
            transformed_points = point_cloud

        normalized_points = transformed_points / self.depth

        pixel_x = ((normalized_points[:, :, 0] + 1) / 2 * self.resolution).clamp(0, self.resolution - 1)
        pixel_y = ((1 - normalized_points[:, :, 1]) / 2 * self.resolution).clamp(0, self.resolution - 1)

        depth_map = torch.zeros((batch_size, self.resolution, self.resolution), dtype=torch.float32, device=point_cloud.device)

        for i in range(batch_size):
            for j in range(num_points):
                x = int(pixel_x[i, j])
                y = int(pixel_y[i, j])
                depth = normalized_points[i, j, 2]
                depth_map[i, y, x] = depth

        depth_map = F.interpolate(depth_map.unsqueeze(1), size=(self.resolution, self.resolution), mode='bilinear', align_corners=False)
        depth_map = depth_map.squeeze(1)

        return depth_map

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_points', type=int, default=1024, help='number of points in each input point cloud')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder to save results')
    parser.add_argument('--model', type=str, default='', help='path to load a pre-trained model')
    parser.add_argument('--feature_transform', default= 'True' , action='store_true', help='use feature transform')
    parser.add_argument('--manualSeed', type=int, default = 2, help='random seed')
    parser.add_argument('--dataset_path', type=str, default= 'dataset/modelnet_scanobjectnn/', help="dataset path")
    parser.add_argument('--ntasks', type=str, default= '1', help="number of tasks")
    parser.add_argument('--nclasses', type=str, default= '26', help="number of classes")
    parser.add_argument('--task', type=str, default= '0', help="task number")
    parser.add_argument('--num_samples', type=str, default= '0', help="number of samples per class")

    opt = parser.parse_args()

    resolution = 64
    depth = 1.5

    path = Path(opt.dataset_path)

    dataloader = DatasetGen(opt, root=path, fewshot=argument.fewshot)
    t = 0
    dataset = dataloader.get(t,'training')
    trainloader = dataset[t]['train']
    testloader = dataset[t]['test'] 

    # sample data from train datalaoder
    data = next(iter(trainloader))
    batch_size = 32
    proj = Realistic_Projection_random()
    clip_model, clip_preprocess = load_clip_model()

    point_cloud, target = data['pointclouds'].float(), data['labels']

    # forward to Realistic Projection
    features_2D_var = torch.zeros((20000, 512))
    for w in range(10):
        pc_prj = proj.get_img(point_cloud[w,:,:].unsqueeze(0))

        features_2D = torch.zeros((200, 512))
        with torch.no_grad():
            pc_img = torch.nn.functional.interpolate(pc_prj, size=(224, 224), mode='bilinear', align_corners=True)
            features_2D = clip_model.encode_image(pc_img)
            features_2D_var[w*200:(w+1)*200,:] = features_2D
        

        

    # use tsne to visualize the features
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=0)
    features_2D_var = tsne.fit_transform(features_2D_var)
    print(features_2D_var.shape)
    # plot the result for the first 200 points with read color and the last 200 points with blue color
    plt.scatter(features_2D_var[0:200,0], features_2D_var[0:200,1], c='r')
    plt.scatter(features_2D_var[200:400,0], features_2D_var[200:400,1], c='b')
    plt.scatter(features_2D_var[400:600,0], features_2D_var[400:600,1], c='g')
    plt.scatter(features_2D_var[600:800,0], features_2D_var[600:800,1], c='y')
    plt.scatter(features_2D_var[800:1000,0], features_2D_var[800:1000,1], c='m')
    plt.scatter(features_2D_var[1000:1200,0], features_2D_var[1000:1200,1], c='c')
    plt.scatter(features_2D_var[1200:1400,0], features_2D_var[1200:1400,1], c='k')
    plt.scatter(features_2D_var[1400:1600,0], features_2D_var[1400:1600,1], c='tab:orange')
    plt.scatter(features_2D_var[1600:1800,0], features_2D_var[1600:1800,1], c='tab:purple')
    plt.scatter(features_2D_var[1800:2000,0], features_2D_var[1800:2000,1], c='tab:brown')
  
    plt.show()
    print(features_2D_var.shape)


