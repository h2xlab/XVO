import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from params import par
import os

def get_test_data_info(data_testing, data_pth):
    
    imgs_path = []

    for key in data_testing.keys():
        for scene in data_testing[key]:
            scene_imgs_path = glob.glob(f'{data_pth[key]}/sequences/{scene}/image_2/*.jpg') + glob.glob(f'{data_pth[key]}/sequences/{scene}/image_2/*.png')
            scene_imgs_path = sorted(scene_imgs_path)
            scene_imgs_path_paired = [[scene_imgs_path[i], scene_imgs_path[i+1]] for i in range(len(scene_imgs_path)-1)]
            imgs_path = imgs_path + scene_imgs_path_paired
    data = {'imgs_path': imgs_path}
    df = pd.DataFrame(data, columns = ['imgs_path'])

    return df

class TestDataset(Dataset):

    def __init__(self, data_df, img_size):

        self.data_df = data_df
        self.imgs_path = np.asarray(self.data_df.imgs_path)

        transform_ops = []
        transform_ops.append(transforms.Resize((img_size[0], img_size[1]), interpolation=transforms.InterpolationMode.BICUBIC))
        transform_ops.append(transforms.ToTensor())
        self.transformer = transforms.Compose(transform_ops)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pth = self.imgs_path[idx]
        img1 = Image.open(pth[0])
        img2 = Image.open(pth[1])
        w1, h1 = img1.size
        w2, h2 = img2.size
        assert w1 == w2
        assert h1 == h2

        # We do crop is because KITTI dataset is a bit old, img size is around 1241*376
        k = 1
        h = 0
        w = (184+384)/2
        img1 = transforms.functional.crop(img1, h, w, 370*k, 658*k)
        img2 = transforms.functional.crop(img2, h, w, 370*k, 658*k)

        img1 = self.transformer(img1)
        img2 = self.transformer(img2)   
        imgs = torch.cat((img1, img2), 0)

        return imgs
