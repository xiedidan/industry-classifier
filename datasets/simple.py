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
            classes.sort()
            self.num_classes = len(classes)

            for item in classes:
                class_path = os.path.join(self.root, self.phase, item)

                pics = os.listdir(class_path)
                pic_paths = [os.path.join(class_path, pic) for pic in pics]
                pic_paths.sort()
                self.samples.append([(pic_path, int(item)) for pic_path in pic_paths])
            
            # for data balance between classes
            class_counts = [len(class_sample) for class_sample in self.samples]
            class_counts.sort()
            self.max_class_count = class_counts[-1]
            
            self.total_len = self.max_class_count * self.num_classes
        else: # test
            classes = os.listdir(os.path.join(self.root, self.phase))
            classes.sort()

            for item in classes:
                class_path = os.path.join(self.root, self.phase, item)

                pics = os.listdir(class_path)
                pic_paths = [os.path.join(class_path, pic) for pic in pics]
                pic_paths.sort()

                self.samples.append([(pic_path, int(item)) for pic_path in pic_paths])
                self.total_len += len(pic_paths)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        if (self.phase == 'train') or (self.phase == 'val'):
            class_index = index // self.max_class_count
            item_index = (index % self.max_class_count) % len(self.samples[class_index])

            image_path, gt = self.samples[class_index][item_index]

            image = Image.open(image_path)
            
            if self.transform is not None:
                image = self.transform(image)

            return image, gt
        else: # test
            total_count = 0

            for class_index, item in enumerate(self.samples):
                if total_count <= index < total_count + len(item):
                    item_index = index - total_count

                    sample = item[item_index]
                    image_path, gt = sample

                    image = Image.open(image_path)

                    if self.transform is not None:
                        image = self.transform(image)

                    return image_path, image, gt
                else:
                    total_count += len(item)
