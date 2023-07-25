# Clustering using CLIP Features

This repository contains a Python script for clustering point cloud data using CLIP (Contrastive Language-Image Pretraining) features. CLIP is a powerful vision-and-language pretraining model developed by OpenAI.

## Setup and Dependencies

To run the script, you will need the following dependencies:

- `numpy`
- `torch`
- `torchvision`
- `PIL` (Python Imaging Library)
- `scikit-learn`

You can install the required dependencies using `pip`:

```
pip install numpy torch torchvision Pillow scikit-learn
```

Additionally, the script requires the following modules which are already included in the repository:

- `open_clip`: A module for creating and using the CLIP model.
- `utils.mv_utils_zs`: A module for Realistic Projection.

## How to Use

1. Ensure that you have all the dependencies installed.

2. Run the `main()` function in the script to start the clustering process.

   ```
   python script.py
   ```

3. The script will perform the following steps:

   - Load the pre-trained CLIP model with the ViT-B-32 architecture.
   - Load the Realistic Projection object.
   - Generate 100 random point cloud samples using PyTorch.
   - Forward each sample through the CLIP model to get feature vectors.
   - Cluster the feature vectors using KMeans into 10 groups.

## Customization

You can customize the clustering process by modifying the following parameters:

- `n_clusters`: The number of clusters to create. By default, it is set to 10.
- `n_components`: The number of top singular values and vectors to keep for SVD. By default, it is set to 10.

You can also try different pre-trained CLIP models and architectures by modifying the `clip_model()` function. The current implementation uses the 'ViT-B-32' architecture with the 'laion2b_s34b_b79k' pre-trained weights.

## Note

The script uses GPU acceleration if a CUDA-enabled GPU is available; otherwise, it falls back to CPU. If you want to force using the CPU, modify the device variable:

```python
device = torch.device("cpu")
```

## License

This code is provided under the [MIT License](LICENSE). Feel free to use and modify it for your needs.

## Acknowledgements

- The CLIP model is from the [open_clip](https://github.com/openai/CLIP) repository.
- The Realistic Projection module is from the [utils.mv_utils_zs](https://github.com/smartgeometry-ucl/dl2021) repository.

Please consider citing the respective repositories if you use this code in your research or project.

---

Enjoy clustering your point cloud data using CLIP features! If you have any questions or suggestions, feel free to contact us. Happy coding!