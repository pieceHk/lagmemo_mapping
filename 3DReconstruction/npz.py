import numpy as np
import matplotlib.pyplot as plt
import os

# ===== 修改此处为你的 npz 文件路径 =====
npz_path = "/home/sfs/SplaTAM/HaiNing_1/0/w2c.npz"

# 读取 npz 文件
data = np.load(npz_path)

print(f"✅ 成功读取: {npz_path}")
print(f"包含以下数组: {data.files}")
print("-" * 60)

# 遍历每个数组
for key in data.files:
    arr = data[key]
    print(f"🔹 数组名: {key}")
    print(f"   形状 shape: {arr.shape}")
    print(f"   数据类型 dtype: {arr.dtype}")
    print(f"   数值范围: [{arr.min():.4f}, {arr.max():.4f}]")

    # 判断是否适合显示
    if arr.ndim == 2:
        plt.title(f"{key} (2D Image)")
        plt.imshow(arr, cmap='gray')
        plt.colorbar()
        plt.show()
    elif arr.ndim == 3 and arr.shape[2] in [1, 3, 4]:
        plt.title(f"{key} (3D Image)")
        plt.imshow(arr.astype(np.uint8))
        plt.show()

    print("-" * 60)
