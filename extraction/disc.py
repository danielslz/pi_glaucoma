import os
import numpy as np

from scipy.ndimage import binary_fill_holes
from skimage import exposure, io
from skimage.color import rgb2hsv
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.util import invert
from skimage.morphology import disk, opening, convex_hull_image

from core import show_imgs
from core.segmentation import ellipse_fitting, draw_ellipse_fitting


def disc_extraction(img, show_steps=False, return_disc_area=False):
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
    output = img.copy()
    output[img_mask == 0] = (0, 0, 0)

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
            {'data': output, 'title': 'disc extraction'},
        ]

        show_imgs(imgs, cols=5)
    
    if return_disc_area:
        return img_ellipse
    
    return output
