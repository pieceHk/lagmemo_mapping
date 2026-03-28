import numpy as np
frame_ids = ['0257', '0263', '0276']#['0126', '0135', '0091']
for frame in frame_ids:
    datapath = f"/home/sfs/0504data/result/0/depth/img{frame}.npy"
    outpath = f"/home/sfs/0504data/result/0/depth_vis/img{frame}.png"
    depth = np.load(datapath)
    min_depth=0.5
    max_depth=5.0
    depth_normalized = (depth - min_depth) / (max_depth - min_depth)
    # clip截断
    depth_normalized = np.clip(depth_normalized, 0, 1)
    import matplotlib.pyplot as plt
    # 不画坐标
    plt.axis('off')
    plt.imshow(depth_normalized, cmap='gray', vmin=0, vmax=1)
    plt.savefig(outpath,bbox_inches='tight',pad_inches=0)