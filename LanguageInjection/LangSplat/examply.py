from plyfile import PlyData
import numpy as np

# 加载 PLY 文件
ply_data = PlyData.read("/home/sfs/LangSplat/output/640480room_1/point_cloud/iteration_30000/point_cloud.ply")

# 查看参数名
for element in ply_data.elements:
    print(f"Element: {element.name}")
    for prop in element.properties:
        print(f"  Property: {prop.name}")