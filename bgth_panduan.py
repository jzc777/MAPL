import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像（以灰度模式）
image_path = 'D:/python_work/AAAShenDuXueXi/Accf/MAPL/datasets/MVTec/pcb1/test/good/0009.png'

gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 使用Otsu的方法自动找到一个全局阈值
ret_otsu, otsu_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu's threshold: {ret_otsu}")

# 使用自适应阈值法
adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

# 可视化结果
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(gray_image, cmap='gray')
plt.subplot(2, 2, 2)
plt.title("Histogram")
plt.hist(gray_image.ravel(), 256, [0,256])
plt.axvline(x=ret_otsu, color='r', linestyle='dashed', linewidth=2)
plt.subplot(2, 2, 3)
plt.title("Otsu's Thresholding")
plt.imshow(otsu_thresh, cmap='gray')
plt.subplot(2, 2, 4)
plt.title("Adaptive Thresholding")
plt.imshow(adaptive_thresh, cmap='gray')
plt.show()
