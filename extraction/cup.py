import os
import numpy as np

from scipy.ndimage import binary_fill_holes
from skimage import exposure, io, img_as_ubyte
from skimage.feature import canny
from skimage.util import invert
from skimage.morphology import disk, opening, convex_hull_image

from core import show_imgs
from core.segmentation import region_growing, ellipse_fitting, draw_ellipse_fitting


def cup_extraction(img, show_steps=False):
    p2, p98 = np.percentile(img, (2, 98))
    
    # original contrast stretching
    img_contrast = exposure.rescale_intensity(img, in_range=(p2, p98))

    # separate green channel
    g_img = img_contrast[:, :, 1]
    
    # contrast stretching green channel
    g_img_rescale = exposure.rescale_intensity(g_img, in_range=(p2, p98))

    # negative transform
    g_img_inverted = invert(g_img_rescale)

    # opening
    g_img_opening = opening(g_img_inverted, disk(40))  # disk(40)

    # negative transform
    g_img_inverted_2 = invert(g_img_opening)

    # region grow
    seed = np.unravel_index(g_img_inverted_2.argmax(), g_img_inverted_2.shape)
    g_img_region_grow = region_growing(g_img_inverted_2, seed, threshold=35)

    # convex hull
    g_img_convex_hull = convex_hull_image(g_img_region_grow)

    # ellipse fitting
    g_img_ellipse = ellipse_fitting(g_img_convex_hull)

    # canny edge detector
    g_img_canny = canny(g_img_ellipse)

    # cup mask
    g_img_mask = binary_fill_holes(g_img_canny)
    
    # cup on original
    g_img_cup = draw_ellipse_fitting(img, g_img_ellipse)

    # # extract cup
    # output = img.copy()
    # output[g_img_mask == 0] = (0, 0, 0)

    if show_steps:
        imgs = [
            {'data': img, 'title': 'original'},
            {'data': img_contrast, 'title': 'contrast stretching'},
            {'data': g_img, 'title': 'green channel'},
            {'data': g_img_rescale, 'title': 'contrast stretching'},
            {'data': g_img_inverted, 'title': 'negative transform'},
            {'data': g_img_opening, 'title': 'opening'},
            {'data': g_img_inverted_2, 'title': 'negative transform'},
            {'data': g_img_region_grow, 'title': 'region grow'},
            {'data': g_img_convex_hull, 'title': 'convex hull'},
            {'data': g_img_ellipse, 'title': 'ellipse fitting'},
            {'data': g_img_canny, 'title': 'canny edge detector'},
            {'data': g_img_mask, 'title': 'mask'},
            {'data': g_img_cup, 'title': 'cup detection'},
            # {'data': output, 'title': 'cup extraction'},
        ]

        show_imgs(imgs, cols=5)
    
    return g_img_ellipse


def bulk_cup_extraction(src, dest):
    # create dest if not exist
    if not os.path.exists(dest):
        os.makedirs(dest)

    with os.scandir(src) as entries:
        for entry in entries:
            if entry.is_file() and not entry.name.startswith('.'):
                print(f'Extracting cup of {entry.name}')
                # convert to np.array
                img = io.imread(src + entry.name)
                # cup extraction
                new_img = cup_extraction(img)
                # save on dest
                io.imsave(dest + entry.name, img_as_ubyte(new_img))
