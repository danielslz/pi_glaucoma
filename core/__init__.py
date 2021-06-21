import matplotlib.pyplot as plt
import numpy as np
import os

from math import ceil
from skimage import io


def show_img(img):
    plt.imshow(img)
    plt.show()


def split_img(img):
    xs = img.shape[0]
    ys = img.shape[1]//2
    return img[0:xs, 0:ys]


def extract_roi(img):
    h = img.shape[0]
    w = img.shape[1]
    crop_h = int(h * 0.3)
    crop_w = int(w * 0.2)
    return img[crop_h:h-crop_h, crop_w:w-crop_w]


def pre_process_imgs(src, dest, filetype='jpg'):
    # create dest if not exist
    if not os.path.exists(dest):
        os.makedirs(dest)

    # list files in src
    with os.scandir(src) as entries:
        for entry in entries:
            if entry.is_file() and not entry.name.startswith('.'):
                print(f'Pre-processing {entry.name}')
                # convert to np.array
                img = io.imread(src + entry.name)
                # split
                s_img = split_img(img)
                # extract roi
                new_img = extract_roi(s_img)
                # save on dest
                io.imsave(dest + entry.name, new_img)


def show_imgs(imgs, cols=3, width=15, height=3, font_size=15):
    if not imgs:
        print("No images to show.")
        return 

    rows = int(ceil(len(imgs) / float(cols)))
    plt.figure(figsize=(width, height*rows), tight_layout=True)

    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i + 1)
        if len(img['data'].shape) == 2:
            # grayscale
            plt.imshow(img['data'], cmap='gray')
        else:
            plt.imshow(img['data'])
        plt.title(img['title'], fontsize=font_size)
        plt.axis(False)
    plt.show()
