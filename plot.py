# -*- coding: utf-8 -*-
from __future__ import print_function

import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def plot_leakage(image_path, result):
    image = Image.open(image_path)
    image = transforms.functional.vflip(image)
    image = transforms.functional.to_tensor(image).numpy()
    image = np.transpose(image, (2, 1, 0))

    color = 'red'
    if result == 0:
        result = u'OK'
        color = 'lime'
    else:
        result = u'Leak'

    plt.ion()
    
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    plt.imshow(image)

    # plt.text(100, 500, result, color=color, fontsize=96)
    plt.xlabel(result, color=color, fontsize=96)

    plt.tight_layout()
    plt.ioff()

    plt.show()
