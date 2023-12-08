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

    def __init__(self, paths, labeled=0, video_len=22, stage='train'):
        self.paths = paths
        self.video_len = video_len
#         self.transform = transform
        self.labeled = labeled
    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        for video in self.paths:
            imgs = self.readImgs(video)
            if (not self.labeled):
                yield imgs
            else:
                msk = self.readMsk(video)
                yield imgs, msk

    def readImgs(self, video):
        # get video frames (22 images)
        imgs = []
        for i in range(self.video_len):
            img = iio.imread(video+('/image_%d.png'%(i)))
#             self.transforms(img)
            imgs.append(img)
        imgs = np.asarray(imgs)
        imgs = torch.tensor(imgs)
        imgs = imgs / 255.0

        return imgs # [B, L, H, W, C] 


    def readMsk(self, video):
        # get mask
        msk = np.load(video+"/mask.npyc")
        msk = torch.tensor(msk) 
        return msk # [B, L, H, W] 
    
    
