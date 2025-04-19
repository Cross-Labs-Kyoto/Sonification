#!/usr/bin/env python3
from pathlib import Path
from torch import nn
from torchvision.models import vgg11 as vgg, VGG11_Weights as Weights
from torchvision.io import decode_image
from matplotlib import pyplot as plt


# Define the root directory
ROOT_DIR = Path(__file__).parent


# Define a function to get the activation values of a layer
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach().squeeze(0).numpy()
    return hook

# Instantiate a vgg model and its corresponding transform
transforms = Weights.DEFAULT.transforms()
model = vgg(weights=Weights.DEFAULT)
model.eval()

# Extract filters and biases
filters = []
biases = []
for name, chd in model.features.named_children():
    if isinstance(chd, nn.Conv2d):
        # Apply the forward hook
        chd.register_forward_hook(get_activation(name))

        # Store the filters and biases
        for p in chd.parameters():
            if p.dim() > 1:
                filters.append(p.detach().permute(0, 2, 3, 1).numpy())
            else:
                biases.append(p.detach().numpy())

# Display filters of first layer
lay_filters = filters[0]
f_max, f_min = lay_filters.max(), lay_filters.min()
norm_filters = (lay_filters - f_min) / (f_max - f_min)

num_filters = lay_filters.shape[0]
for f_idx in range(num_filters):
    f = norm_filters[f_idx]
    ax = plt.subplot(num_filters // 8, 8, f_idx + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(norm_filters[f_idx])
plt.show()

# Visualize feature maps
# Load the image from the Data directory
img_path = ROOT_DIR.joinpath('Data', 'bird.jpg')
img_tensor = decode_image(img_path, mode='RGB')

# Forward image through model
t_img = transforms(img_tensor.unsqueeze(0))
model(t_img)

for blk_idx, feat_maps in activation.items():
    num_maps = feat_maps.shape[0]
    for f_idx in range(num_maps):
        ax = plt.subplot(num_maps // 16, 16, f_idx + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(feat_maps[f_idx], cmap='inferno')
    fig = plt.gcf()
    fig.set_size_inches(w=22, h=16)
    plt.savefig(ROOT_DIR.joinpath('Data', f'featmap_block_{blk_idx}.png'), format='png')
