# PointNet Classifier with CLIP-based Features

This repository contains a PointNet classifier with CLIP-based features. The PointNet classifier is trained to classify 3D point clouds into one of the predefined classes, using the CLIP model to extract image features from the point clouds.

## Prerequisites

Before running the code, make sure you have the following prerequisites installed:

- Python 3.x
- PyTorch
- TorchVision
- NumPy
- tqdm
- OpenCLIP

Install the required dependencies using `pip`:

```bash
pip install torch torchvision numpy tqdm open_clip
```

## Getting Started

Clone the repository and navigate to the project directory:

```bash
git clone <repository_url>
cd <repository_directory>
```

## Usage

To train the PointNet classifier with CLIP-based features, use the following command:

```bash
python train.py
```

## Command-line Arguments

The script `train.py` supports several command-line arguments to customize the training process. The available arguments are as follows:

- `--batchSize`: The batch size for training (default: 32).
- `--num_points`: The number of points in each input point cloud (default: 1024).
- `--workers`: The number of data loading workers (default: 4).
- `--nepoch`: The number of epochs to train for (default: 250).
- `--outf`: The output folder to save the training results (default: 'cls').
- `--model`: Path to load a pre-trained model (default: '').
- `--feature_transform`: Use feature transform (default: False).

## Results

The trained models will be saved in the specified output folder (`outf`) with the name `cls_model_<epoch>.pth`, where `<epoch>` represents the epoch number.

After training, the final accuracy on the test set will be displayed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The PointNet model is based on the implementation from [PointNet PyTorch](https://github.com/fxia22/pointnet.pytorch).
- The CLIP model and preprocessing functions are from the [OpenCLIP](https://github.com/openai/CLIP) library.
- The Realistic Projection object is provided by `Realistic_Projection` in `utils.mv_utils_zs` module.

## Contact

For any questions or inquiries, please contact [author@example.com](mailto:author@example.com).
