
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from torch.utils.data import random_split
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import models, transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from fastai.vision.all import show_image


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.model(x)


class conv_transpose_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_transpose_block, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_input, i):
        # print(i)
        x = self.model(x)
        # print(x.shape)
        # print(skip_input.shape)
        x = torch.cat((x, skip_input), 1)
        return x


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNET, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.c1 = conv_block(in_channels, 64)
        self.c2 = conv_block(64, 128)
        self.c3 = conv_block(128, 256)
        self.c4 = conv_block(256, 512)
        self.c5 = conv_block(512, 1024)
        self.ct1 = conv_transpose_block(1024, 512)
        self.c6 = conv_block(1024, 512)
        self.ct2 = conv_transpose_block(512, 256)
        self.c7 = conv_block(512, 256)
        self.ct3 = conv_transpose_block(256, 128)
        self.c8 = conv_block(256, 128)
        self.ct4 = conv_transpose_block(128, 64)
        self.c9 = conv_block(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        l1 = self.c1(x)  # channels:64
        l2 = self.pool(l1)  # size:256
        l3 = self.c2(l2)  # channels:128
        l4 = self.pool(l3)  # size:128
        l5 = self.c3(l4)  # channels:256
        l6 = self.pool(l5)  # size:64
        l7 = self.c4(l6)  # channels:512
        l8 = self.pool(l7)  # size:32
        l9 = self.c5(l8)  # channels:1024
        l10 = self.ct1(l9, l7, 1)  # size:64
        l11 = self.c6(l10)  # channels:512
        l12 = self.ct2(l11, l5, 2)  # size:128
        l13 = self.c7(l12)  # channels:256
        l14 = self.ct3(l13, l3, 3)  # size:256
        l15 = self.c8(l14)  # channels:128
        l16 = self.ct4(l15, l1, 4)  # size:512
        l17 = self.c9(l16)  # channels:64
        l18 = self.final(l17)  # size:512 channels:3
        return l18


FILE = "segmentation_model_200.pth"
model = UNET(in_channels=3, out_channels=3).to('cuda')
model.load_state_dict(torch.load(FILE))
model.eval()

data_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize((256, 512)),
    transforms.Normalize(mean=[0.0, 0.0, 0.0],
                         std=[1.0, 1.0, 1.0]),

])
batch_size = 1
data_dir = r"D:\dataset\cityscapes\cityscapes\val"
val = ImageFolder(data_dir, transform=data_transform)
val_dl = DataLoader(val, batch_size, num_workers=4, pin_memory=True)
len(val_dl)

Normalization_Values = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)


def DeNormalize(tensor_of_image):
    return tensor_of_image * Normalization_Values[1][0] + Normalization_Values[0][0]


vidObj = cv2.VideoCapture(r"D:\video\a.mp4")
# vidObj = cv2.VideoCapture(r"D:\videos\output_video.avi")
# Used as counter variable
count = 0

# checks whether frames were extracted
success = 1
data_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.0, 0.0, 0.0],
                         std=[1.0, 1.0, 1.0]),

])
while success:

    # vidObj object calls read
    # function extract frames
    success, image = vidObj.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image = data_transform(image)
    image = image.unsqueeze(0).permute(0, 1, 3, 2).to('cuda')
    # print(image)
    # print(image.shape)
    # Saves the frames with frame-count
    pred2 = model(image)
    images = DeNormalize(pred2)
    print(images.shape)
    images = images.detach().cpu()
    images = images[0].numpy().transpose(1, 2, 0)
    images = cv2.resize(images, (512, 512))
    # print(images[0])
    # image_grid = make_grid(images[:5], nrow=5)
    # plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    # plt.show()
    cv2.imshow('pred', images)
    cv2.imshow('target', image)
    if cv2.waitKey(0) & 0xFF == ord('c'):
        cv2.destroyAllWindows
