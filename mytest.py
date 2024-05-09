import os
import PIL.Image
import numpy as np

#
def convert_2d(r):
    # 添加均值为 0, 标准差为 16 的加性高斯白噪声
    s = r + np.random.normal(0, 16, r.shape)
    if np.min(s) >= 0 and np.max(s) <= 255:
        return s
    # 对比拉伸
    s = s - np.full(s.shape, np.min(s))
    s = s * 255 / np.max(s)
    s = s.astype(np.uint8)
    return s


def convert_3d(r):
    s_dsplit = []
    for d in range(r.shape[2]):
        rr = r[:, :, d]
        ss = convert_2d(rr)
        s_dsplit.append(ss)
    s = np.dstack(s_dsplit)
    return s


def process_images(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpeg"):  # 根据需要调整文件格式
            img_path = os.path.join(folder_path, filename)
            im = PIL.Image.open(img_path)
            im = im.convert('RGB')
            im_mat = np.asarray(im)
            im_converted_mat = convert_3d(im_mat)
            im_converted = PIL.Image.fromarray(im_converted_mat)

            # 保存转换后的图片
            output_path = os.path.join(output_folder, filename)
            im_converted.save(output_path)
            print(f"Processed and saved: {output_path}")


# 使用示例
source_folder = 'E:/ADataset/transistor/test/misplaced'
output_folder = 'D:/python_work/AAAShenDuXueXi/Accf/MAPL/datasets/MVTec/transistor/test/misplaced'
process_images(source_folder, output_folder)




