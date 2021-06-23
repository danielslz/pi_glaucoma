import os
import numpy as np

from scipy.ndimage import binary_fill_holes
from skimage import exposure, io, img_as_ubyte
from skimage.color import rgb2hsv
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.morphology import disk, opening, convex_hull_image, remove_small_objects

from core import show_imgs
from core.segmentation import ellipse_fitting, draw_ellipse_fitting


def disc_extraction(img, show_steps=False):
    p2, p98 = np.percentile(img, (2, 98))
    
    # original contrast stretching
    img_contrast = exposure.rescale_intensity(img, in_range=(p2, p98))

    # rgb to hsv
    img_hsv = rgb2hsv(img_contrast)

    # hsv contrast stretching
    value_img = img_hsv[:, :, 2]
    img_hsv_contrast = exposure.rescale_intensity(value_img)
    
    # mean threshold
    thresh = threshold_otsu(img_hsv_contrast)
    img_mean_threshold = img_hsv_contrast > thresh

    # opening
    img_opening = opening(img_mean_threshold, disk(15))
    img_opening = remove_small_objects(img_opening, 30000)

    # convex hull
    img_convex_hull = convex_hull_image(img_opening)

    # ellipse fitting
    img_ellipse = ellipse_fitting(img_convex_hull)

    # canny edge detector
    img_canny = canny(img_ellipse)

    # disc mask
    img_mask = binary_fill_holes(img_canny)
    
    # disc on original
    img_disc = draw_ellipse_fitting(img, img_ellipse)

    # extract disc
    # output = img.copy()
    # output[img_mask == 0] = (0, 0, 0)

    if show_steps:
        imgs = [
            {'data': img, 'title': 'original'},
            {'data': img_contrast, 'title': 'contrast stretching'},
            {'data': img_hsv, 'title': 'convert to hsv'},
            {'data': img_hsv_contrast, 'title': 'contrast stretching'},
            {'data': img_mean_threshold, 'title': 'otsu threshold'},
            {'data': img_opening, 'title': 'opening'},
            {'data': img_convex_hull, 'title': 'convex hull'},
            {'data': img_ellipse, 'title': 'ellipse fitting'},
            {'data': img_canny, 'title': 'canny edge detector'},
            {'data': img_mask, 'title': 'mask'},
            {'data': img_disc, 'title': 'disc detection'},
            # {'data': output, 'title': 'disc extraction'},
        ]

        show_imgs(imgs, cols=5)
    
    return img_ellipse


def bulk_disc_extraction(src, dest):
    # create dest if not exist
    if not os.path.exists(dest):
        os.makedirs(dest)

    with os.scandir(src) as entries:
        for entry in entries:
            if entry.is_file() and not entry.name.startswith('.'):
                print(f'Extracting disc of {entry.name}')
                # convert to np.array
                img = io.imread(src + entry.name)
                # cup extraction
                new_img = disc_extraction(img)
                # save on dest
                io.imsave(dest + entry.name, img_as_ubyte(new_img))
