# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import os
import pickle
import argparse
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from datasets.simple import *

# config
num_classes = 2
size = 255

batch_size = 8
num_workers = 4

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
# device = 'cpu'

# arg
parser = argparse.ArgumentParser(description='PyTorch VGG Classifier Testing')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint.pth', help='checkpoint file path')
parser.add_argument('--root', default='/media/voyager/ssd-ext4/industry/', help='dataset root path')
flags = parser.parse_args()

# data
testTransform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()
])

testSet = SimpleDataset(
    root=flags.root,
    phase='test',
    transform=testTransform
)

testLoader = DataLoader(
    dataset=testSet,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)

# model
model = models.vgg16_bn(
    False,
    num_classes=num_classes
)
model.to(device)

checkpoint = torch.load(flags.checkpoint)
model.load_state_dict(checkpoint['net'])

# pipeline
def test():
    with torch.no_grad():
        model.eval()

        for batch_index, samples in enumerate(testLoader):
            images, gts = samples

            images = images.to(device)

            output = model(images)
            output = F.softmax(output, dim=len(output.size())-1)
            output = torch.argmax(
                output,
                dim=len(output.size())-1,
                keepdim=False
            )

            print('\nbatch: {}\noutput: {}\ngt: {}'.format(batch_index, output, gts))

# main
if __name__ == '__main__':
    test()
