from skimage.util import random_noise #添加噪声模块
import matplotlib.pyplot as plt

# image=plt.imread('/data/Disk_A/yongbiao/USERPROG/PIAFusion/test_data/MSRS/Vis/00537D.png')
# print(type(image))  #图片种类
# print(image.shape)  #打印图片大小
# print(image.dtype)  #打印图片数据类型
# plt.subplot(2,2,1)
# plt.title('origin')
# plt.imshow(image)
# import numpy as np
#
# noise_gaussian_1=random_noise(image,mode="gaussian",clip=True)
# noise_gaussian_2=random_noise(image,mode="gaussian",var=0.1,clip=True)
# noise_gaussian_3=random_noise(image,mode="gaussian",var=1,clip=True)
#
# import imageio
# imageio.imsave('test.jpg', noise_gaussian_1)
#
#
# from PIL import Image
# X = Image.fromarray(noise_gaussian_1)
#
# X.save("1.png")
# noise_gaussian_3.save('1.png')



# plt.subplot(2,2,2)
# plt.title('var=0.01')
# plt.imshow(noise_gaussian_1)
# plt.subplot(2,2,3)
# plt.title('var=0.1')
# plt.imshow(noise_gaussian_2)
# plt.subplot(2,2,4)
# plt.title('var=1')
# plt.imshow(noise_gaussian_3)
# plt.show()

import argparse
import random

import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from loader.msrs_data import MSRS_data
from test_model_3.model_conv4 import WaveEncoder, WaveDecoder
from PIL import Image
from torch.utils import data
from torchvision import transforms
import fusion_strategy
import torch
from time import time
test_dataset = MSRS_data('/data/Disk_A/yongbiao/USERPROG/PIAFusion/test_data/MSRS-main')
test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)

test_tqdm = tqdm(test_loader, total=len(test_loader))

for vis_image,  inf_image, name, size in test_tqdm:


    noise_gaussian_1 = random_noise(vis_image, mode="gaussian", var=0.5,clip=True)
    import imageio
    imageio.imsave(f'/data/Disk_A/yongbiao/USERPROG/PIAFusion/test_data/MSRS_noise_0.5/Vis/{name[0]}', noise_gaussian_1[0])


    # fused_image = YCrCb2RGB(fused_image, cb[0], cr[0])
    # fused_image = transforms.ToPILImage()(fused_image)
    # fused_image=fused_image.resize((size[0],size[1]),Image.BILINEAR)
    # fused_image.save(f'{args.save_path}/{name[0]}')