# @Tanya Wang 

import os
import torch
import torchvision
from torchvision import transforms
# from PIL import Image # PIL is a library to process images
from matplotlib import pyplot as plt
import glob
from glob import glob
import numpy as np
import imageio.v3 as iio

simple_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                ])

class VideoDataset(torch.utils.data.IterableDataset):

    def __init__(self, paths, train=False, video_len=22, stage='train'):
        self.paths = paths
        self.video_len = video_len
#         self.transform = transform
        self.train = train 

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        for video in self.paths:
            imgs_x = self.readImgs(video, range(self.video_len))
            if not self.train: 
                yield imgs_x 
            else:
                imgs_y = self.readImgs(video, range(self.video_len, 2*self.video_len))
                yield imgs_x, imgs_y 

    def readImgs(self, video, load_range):
        # get video frames (22 images)
        imgs = []
        for i in load_range:
            img = iio.imread(video+('/image_%d.png'%(i)))
#             self.transforms(img)
            imgs.append(img)
        imgs = np.asarray(imgs)
        imgs = torch.tensor(imgs)
        imgs = imgs / 255.0 # [B, L, H, W, C] 

        imgs = imgs.permute(0, 1, 4, 2, 3) # [B, L, C, H, W] 

        return imgs 

    
