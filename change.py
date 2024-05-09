import cv2
import os
import numpy as np

#生成掩码
def process_and_save_images(source_folder, target_folder):
    # 确保目标文件夹存在，如果不存在，则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if filename.endswith(".png"):  # 检查文件扩展名
            # 构建完整的文件路径
            file_path = os.path.join(source_folder, filename)

            # 读取图像（以灰度模式）
            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            # 将所有非零值转换为255（白色）
            mask[mask > 0] = 255

            # 构建目标文件的完整路径
            target_path = os.path.join(target_folder, filename)

            # 保存修改后的图像到目标文件夹
            cv2.imwrite(target_path, mask)


# 源文件夹路径
source_folder = 'D:/python_work/AAAShenDuXueXi/Accf/MAPL/datasets/MVTec/pcb1/ground_truth/Anomaly'
# 目标文件夹路径
target_folder = 'D:/python_work/AAAShenDuXueXi/Accf/MAPL/datasets/MVTec/pcb1/ground_truth/A1'

# 调用函数处理并保存图像
process_and_save_images(source_folder, target_folder)
