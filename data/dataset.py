import cv2
import os
import numpy as np
from PIL import Image
from glob import glob
from einops import rearrange

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import imgaug.augmenters as iaa

from data import rand_perlin_2d_np

from typing import List

import cv2
import os
import numpy as np
from glob import glob
from einops import rearrange

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import imgaug.augmenters as iaa

from .perlin import rand_perlin_2d_np

from typing import List

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MAPLDataset(Dataset):
    def __init__(
            self, datadir: str, target: str, is_train: bool, to_memory: bool = False,
            resize: List[int] = [256, 256], imagesize: int = 224,
            texture_source_dir: str = None, structure_grid_size: str = 8,
            transparency_range: List[float] = [0.15, 1.],
            perlin_scale: int = 6, min_perlin_scale: int = 0, perlin_noise_threshold: float = 0.5,
            use_mask: bool = True, bg_threshold: float = 100, bg_reverse: bool = False
    ):
        # mode
        self.is_train = is_train
        self.to_memory = to_memory

        # load image file list
        self.datadir = datadir
        self.target = target
        self.file_list = glob(os.path.join(self.datadir, self.target, 'train/*/*' if is_train else 'test/*/*'))

        # synthetic anomaly
        if self.is_train and not self.to_memory:
            # load texture image file list
            self.texture_source_file_list = glob(
                os.path.join(texture_source_dir, '*/*')) if texture_source_dir else None

            # perlin noise
            self.perlin_scale = perlin_scale
            self.min_perlin_scale = min_perlin_scale
            self.perlin_noise_threshold = perlin_noise_threshold

            # structure
            self.structure_grid_size = structure_grid_size

            # anomaly mixing
            self.transparency_range = transparency_range

            # mask setting
            self.use_mask = use_mask
            self.bg_threshold = bg_threshold
            self.bg_reverse = bg_reverse

        # transform
        self.resize = list(resize)
        self.transform_img = [
            transforms.ToPILImage(),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.ToPILImage(),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        # sythetic anomaly switch
        self.anomaly_switch = False
        self.original_len = len(self.file_list)  # 原始有标签数据的数量
        # 调整后的数据集长度
        if self.is_train and not self.to_memory:
            self.adjusted_len = int(self.original_len * 2.5)  # 为了实现1:1.5的比例
        self.adjusted_len = self.original_len

    def __getitem__(self, idx):
        # 调整索引以映射到原始数据集
        if self.to_memory:
            adjusted_idx = idx
        else:
            adjusted_idx = int(idx // 2.5) if self.is_train else idx

        file_path = self.file_list[adjusted_idx]
        img = Image.open(file_path).convert("RGB").resize(self.resize)
        img = np.array(img)
        target = 0 if 'good' in file_path else 1

        if 'good' in file_path:
            mask = np.zeros(self.resize, dtype=np.float32)
        else:
            mask_path = file_path.replace('test', 'ground_truth').replace('.png', '_mask.png')
            mask = Image.open(mask_path).resize(self.resize)
            mask = np.array(mask)

        # 根据索引确定是返回有标签数据还是生成无标签数据
        if self.is_train and not self.to_memory:
            if idx % 5 != 0 and idx % 5 != 3:
                # 生成无标签数据
                # 请在这里添加生成无标签（异常模拟）数据的具体逻辑
                # 以下代码仅为示例，应根据实际情况调整
                file_path = self.file_list[adjusted_idx % len(self.file_list)]  # 循环使用有标签样本
                img, mask = self.generate_anomaly(img=np.array(Image.open(file_path).convert("RGB").resize(self.resize)),
                                                texture_img_list=self.texture_source_file_list)
                target = 1  # 假设无标签数据为异常数据
                mask = torch.Tensor(mask)

        img = self.transform_img(img)
        mask = self.transform_mask(mask).squeeze()
        # mask = torch.Tensor(mask) if isinstance(mask, np.ndarray) else self.transform_mask(mask).squeeze()

        return img, mask, target

    def rand_augment(self):
        augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

        aug_idx = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
        aug = iaa.Sequential([
            augmenters[aug_idx[0]],
            augmenters[aug_idx[1]],
            augmenters[aug_idx[2]]
        ])

        return aug

    def generate_anomaly(self, img: np.ndarray, texture_img_list: list = None) -> List[np.ndarray]:
        '''
        step 1. generate mask
            - target foreground mask
            - perlin noise mask

        step 2. generate texture or structure anomaly
            - texture: load DTD
            - structure: we first perform random adjustment of mirror symmetry, rotation, brightness, saturation,
            and hue on the input image  𝐼 . Then the preliminary processed image is uniformly divided into a 4×8 grid
            and randomly arranged to obtain the disordered image  𝐼

        step 3. blending image and anomaly source
        '''

        # step 1. generate mask
        img_size = img.shape[:-1]  # H x W

        ## target foreground mask
        if self.use_mask:
            target_foreground_mask = self.generate_target_foreground_mask(img=img)
        else:
            target_foreground_mask = np.ones(self.resize)

        ## perlin noise mask
        perlin_noise_mask = self.generate_perlin_noise_mask(img_size=img_size)

        ## mask
        mask = perlin_noise_mask * target_foreground_mask
        mask_expanded = np.expand_dims(mask, axis=2)

        # step 2. generate texture or structure anomaly

        ## anomaly source
        anomaly_source_img = self.anomaly_source(img=img, texture_img_list=texture_img_list)

        ## mask anomaly parts
        factor = np.random.uniform(*self.transparency_range, size=1)[0]
        anomaly_source_img = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * img)

        # step 3. blending image and anomaly source
        anomaly_source_img = ((- mask_expanded + 1) * img) + anomaly_source_img

        return (anomaly_source_img.astype(np.uint8), mask)

    def generate_target_foreground_mask(self, img: np.ndarray) -> np.ndarray:
        # convert RGB into GRAY scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # generate binary mask of gray scale image
        _, target_background_mask = cv2.threshold(img_gray, self.bg_threshold, 255, cv2.THRESH_BINARY)
        target_background_mask = target_background_mask.astype(np.bool_).astype(np.int32)

        # invert mask for foreground mask
        if self.bg_reverse:
            target_foreground_mask = target_background_mask
        else:
            target_foreground_mask = -(target_background_mask - 1)

        return target_foreground_mask

    def generate_perlin_noise_mask(self, img_size: tuple) -> np.ndarray:
        # define perlin noise scale
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])

        # generate perlin noise
        perlin_noise = rand_perlin_2d_np(img_size, (perlin_scalex, perlin_scaley))

        # apply affine transform
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)

        # make a mask by applying threshold
        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold,
            np.ones_like(perlin_noise),
            np.zeros_like(perlin_noise)
        )

        return mask_noise

    def anomaly_source(self, img: np.ndarray, texture_img_list: list = None) -> np.ndarray:
        p = np.random.uniform() if texture_img_list else 1.0
        if p < 0.5:
            idx = np.random.choice(len(texture_img_list))
            img_size = img.shape[:-1]  # H x W
            anomaly_source_img = self._texture_source(img_size=img_size, texture_img_path=texture_img_list[idx])
        else:
            anomaly_source_img = self._structure_source(img=img)

        return anomaly_source_img

    def _texture_source(self, img_size: tuple, texture_img_path: str) -> np.ndarray:
        texture_source_img = cv2.imread(texture_img_path)
        texture_source_img = cv2.cvtColor(texture_source_img, cv2.COLOR_BGR2RGB)
        texture_source_img = cv2.resize(texture_source_img, dsize=img_size).astype(np.float32)

        return texture_source_img

    def _structure_source(self, img: np.ndarray) -> np.ndarray:
        structure_source_img = self.rand_augment()(image=img)

        img_size = img.shape[:-1]  # H x W

        assert img_size[0] % self.structure_grid_size == 0, 'structure should be devided by grid size accurately'
        grid_w = img_size[1] // self.structure_grid_size
        grid_h = img_size[0] // self.structure_grid_size

        structure_source_img = rearrange(
            tensor=structure_source_img,
            pattern='(h gh) (w gw) c -> (h w) gw gh c',
            gw=grid_w,
            gh=grid_h
        )
        disordered_idx = np.arange(structure_source_img.shape[0])
        np.random.shuffle(disordered_idx)

        structure_source_img = rearrange(
            tensor=structure_source_img[disordered_idx],
            pattern='(h w) gw gh c -> (h gh) (w gw) c',
            h=self.structure_grid_size,
            w=self.structure_grid_size
        ).astype(np.float32)

        return structure_source_img

    def __len__(self):
        # 返回调整后的数据集长度
        return self.adjusted_len


