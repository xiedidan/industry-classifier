'''
original: https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py
'''

import argparse

import torch
import torchvision.transforms as transforms

from datasets.simple import *

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2
    )
    mean = torch.zeros(3)
    std = torch.zeros(3)

    print('Computing mean and std..')

    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()

    mean.div_(len(dataset))
    std.div_(len(dataset))

    return mean, std

if __name__ == '__main__':
    # arg
    parser = argparse.ArgumentParser(description='PyTorch Classifier Util')
    parser.add_argument('--root', default='/media/voyager/ssd-ext4/industry/', help='dataset root path')
    flags = parser.parse_args()

    trainTransform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainSet = SimpleDataset(
        root=flags.root,
        phase='train',
        transform=trainTransform
    )

    mean, std = get_mean_and_std(trainSet)
    print('\nmean: {}\nstd: {}\n'.format(mean, std))
