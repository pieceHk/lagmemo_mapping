import os
import numpy as np
import matplotlib.pyplot as plt
path_origin = "/home/sfs/LangSplat/lagmemodata/gs_data/language_features"
path_compressed = "/home/sfs/LangSplat/lagmemodata/gs_data/language_features_dim3"
data = np.load(os.path.join(path_origin, "img0260_s.npy"))
# print(np.unique(data[3,:,:]))
print(data.shape)
print(np.min(data[3][data[3]>0]))
print(np.max(data[2]))
print(np.unique(data[0]))
print(np.unique(data[1]))
print(np.unique(data[2]))
print(np.unique(data[3]))
mask_num = int(np.max(data[3]))
# for i in range(4):
#     print(np.sum(data[i]==-1))
# Generate random distinct colors (avoid close hues)
from matplotlib.colors import ListedColormap
rng = np.random.RandomState(42)
colors = rng.random((mask_num, 3))  # Random RGB
colors = np.vstack([[0, 0, 0], colors])  # Add black for -1
cmap_custom = ListedColormap(colors)
n_levels = 4
fig, axes = plt.subplots(1, n_levels, figsize=(20,5))
for i in range(n_levels):
    mask = data[i]
    ax = axes[i]
    im = ax.imshow(mask, cmap=cmap_custom, vmin=-1, vmax=mask_num)
    ax.set_title(f'SAM Mask Level {i}')
    ax.axis('off')
plt.colorbar(im, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1, label='Mask ID')
plt.tight_layout()
plt.savefig('sam_masks.png', bbox_inches='tight', dpi=300)