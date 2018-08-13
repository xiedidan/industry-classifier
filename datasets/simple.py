import torch
from torch.utils.data import *
import torchvision.transforms as transforms

from PIL import Image
import sys
import os

class SimpleDataset(Dataset):
    def __init__(
        self,
        root,
        phase='train',
        transform=None):
        self.root = root
        self.phase = phase
        self.transform = transform

        self.num_classes = 0
        self.samples = []
        self.max_class_count = 0
        self.total_len = 0

        if (self.phase == 'train') or (self.phase == 'val'):
            classes = os.listdir(os.path.join(self.root, self.phase))
            classes = classes.sort()
            self.num_classes = len(classes)

            for item in classes:
                class_path = os.path.join(self.root, self.phase, item)

                pics = os.listdir(class_path)
                pic_paths = [os.path.join(class_path, pic) for pic in pics]
                pic_paths = pic_paths.sort()
                self.samples.append([(pic_path, int(item)) for pic_path in pic_paths])
            
            # for data balance between classes
            class_counts = [len(class_sample) for class_sample in self.samples]
            class_counts = class_counts.sort()
            self.max_class_count = class_counts[-1]
            
            self.total_len = self.max_class_count * self.num_classes
        else: # test
            pics = os.listdir(os.path.join(self.root, self.phase))

            self.total_len = len(pics)
            self.samples = [os.path.join(self.root, self.phase, pic) for pic in pics]

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        if (self.phase == 'train') or (self.phase == 'val'):
            class_index = (index + 1) / self.max_class_count
            item_index = ((index + 1) % self.max_class_count) % len(self.samples[class_index])
            image_path, gt = self.samples[class_index][item_index]

            image = Image.open(image_path)
            
            if self.transform is not None:
                image = self.transform(image)

            return image, gt
        else: # test
            image = self.samples[index]

            if self.transform is not None:
                image = self.transform(image)

            return image
