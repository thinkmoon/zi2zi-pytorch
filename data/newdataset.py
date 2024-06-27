from os import listdir
import os
from os.path import join
import random

from PIL import Image, ImageFilter
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np

from utils.image_processing import read_split_image

class FurtherOptimizedDatasetFromObj(data.Dataset):
    def __init__(self, input_nc=1, augment=False, bold=False, rotate=False, blur=False, start_from=0,data_dir='dir'):
        super(FurtherOptimizedDatasetFromObj, self).__init__()
        self.input_nc = input_nc
        if self.input_nc == 1:
            self.transform = transforms.Normalize(0.5, 0.5)
        elif self.input_nc == 3:
            self.transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            raise ValueError('input_nc should be 1 or 3')
        self.augment = augment
        self.bold = bold
        self.rotate = rotate
        self.blur = blur
        self.start_from = start_from

        self.data_dir = data_dir
        self.samples = []  # 存储样本路径和标签
        
        # 遍历data_dir下的所有文件
        for filename in os.listdir(data_dir):
            # 假设文件名结构为"{label}_xxx.png"
            label, _ = os.path.splitext(filename)[0].split('_')  # 分割并提取标签
            if label.isdigit():  # 确保标签是数字
                path = os.path.join(data_dir, filename)
                self.samples.append((path, int(label)))  # 添加样本路径和整数标签到列表中

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        img_A, img_B = self.process(img_path)
        return label, img_A, img_B

    def __len__(self):
        return len(self.samples)

    def process(self, image_file):
        """
            process byte stream to training data entry
        """
        with Image.open(image_file).convert('L') as img:
            try:
                # 直接在内存中操作，而不是创建新的图像对象
                img_A, img_B = read_split_image(img)
                if self.augment:
                    w, h = img_A.size
                    if self.bold:
                        multiplier = random.uniform(1.0, 1.2)
                    else:
                        multiplier = random.uniform(1.0, 1.05)
                    nw = int(multiplier * w) + 1
                    nh = int(multiplier * h) + 1

                    # 在原图像上进行操作，避免创建新的放大图像
                    img_A = img_A.resize((nw, nh), Image.LANCZOS)
                    img_B = img_B.resize((nw, nh), Image.LANCZOS)

                    shift_x = random.randint(0, max(nw - w - 1, 0))
                    shift_y = random.randint(0, max(nh - h - 1, 0))

                    # 直接裁剪原图像，避免复制数据
                    img_A = img_A.crop((shift_x, shift_y, shift_x + w, shift_y + h))
                    img_B = img_B.crop((shift_x, shift_y, shift_x + w, shift_y + h))

                    if self.rotate and random.random() > 0.9:
                        angle_list = [0, 180]
                        random_angle = random.choice(angle_list)
                        if self.input_nc == 3:
                            fill_color = (255, 255, 255)
                        else:
                            fill_color = 255
                        # 旋转原图像，避免复制数据
                        img_A = img_A.rotate(random_angle, resample=Image.BILINEAR, fillcolor=fill_color)
                        img_B = img_B.rotate(random_angle, resample=Image.BILINEAR, fillcolor=fill_color)

                    if self.blur and random.random() > 0.8:
                        sigma_list = [1, 1.5, 2]
                        sigma = random.choice(sigma_list)
                        # 直接对原图像进行模糊处理，避免复制数据
                        img_A = img_A.filter(ImageFilter.GaussianBlur(radius=sigma))
                        img_B = img_B.filter(ImageFilter.GaussianBlur(radius=sigma))

                    # 转换为张量并进行归一化
                    img_A = transforms.ToTensor()(img_A)
                    img_B = transforms.ToTensor()(img_B)
                    img_A = self.transform(img_A)
                    img_B = self.transform(img_B)
                else:
                    img_A = transforms.ToTensor()(img_A)
                    img_B = transforms.ToTensor()(img_B)
                    img_A = self.transform(img_A)
                    img_B = self.transform(img_B)

                return img_A, img_B

            finally:
                # 无需额外操作，with 语句会自动处理资源释放
                pass