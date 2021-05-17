import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image


def show_img(img):
    plt.imshow(img)
    plt.show()


def split_img(img):
    xs = img.shape[0]
    ys = img.shape[1]//2
    return img[0:xs, 0:ys]


def pre_process_imgs(src, dest, filetype='jpg'):
    # list files in src
    with os.scandir(src) as entries:
        for entry in entries:
            if entry.is_file():
                print(f'Processing {entry.name}')
                # convert to np.array
                img = np.asarray(Image.open(src + entry.name))
                # split
                new_img = Image.fromarray(split_img(img))
                # save on dest
                new_img.save(dest + entry.name)
