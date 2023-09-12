
import torch

# Assuming image_embeddings_tmp and image_embeddings are torch tensors

k_values = torch.arange(32)
rows_to_average = k_values.view(-1, 1) + torch.arange(0, 289, 32)  # Generate indices
print(rows_to_average)