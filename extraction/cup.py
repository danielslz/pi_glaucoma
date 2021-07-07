import os
import numpy as np

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
    g_img_opening = opening(g_img_inverted, disk(25))  # disk(40)

    # negative transform
    g_img_inverted_2 = invert(g_img_opening)

    # region grow
    seed = np.unravel_index(g_img.argmax(), g_img.shape)
    g_img_region_grow = region_growing(g_img_inverted_2, seed, threshold=15)  # threshold=25

    # convex hull
    g_img_convex_hull = convex_hull_image(g_img_region_grow)

    # ellipse fitting
    g_img_ellipse = ellipse_fitting(g_img_convex_hull)

    # canny edge detector
    g_img_canny = canny(g_img_ellipse)
    
    # cup on original
    g_img_cup = draw_ellipse_fitting(img, g_img_canny)

    # # extract cup
    # output = img.copy()
    # output[g_img_mask == 0] = (0, 0, 0)

    if show_steps:
        imgs = [
            {'data': img, 'title': 'Original (a)'},
            {'data': img_contrast, 'title': 'Alargamento de contraste (b)'},
            {'data': g_img, 'title': 'Canal Verde (c)'},
            {'data': g_img_rescale, 'title': 'Alargamento de contraste (d)'},
            {'data': g_img_inverted, 'title': 'Transformação negativa (e)'},
            {'data': g_img_opening, 'title': 'Abertura (f)'},
            {'data': g_img_inverted_2, 'title': 'Transformação negativa (g)'},
            {'data': g_img_region_grow, 'title': 'Region growing (h)'},
            {'data': g_img_convex_hull, 'title': 'Fecho convexo (i)'},
            {'data': g_img_ellipse, 'title': 'Preenchimento de elipse (j)'},
            {'data': g_img_canny, 'title': 'Canny (k)'},
            {'data': g_img_cup, 'title': 'Escavação do disco óptico (l)'},
            # {'data': output, 'title': 'cup extraction'},
        ]

        show_imgs(imgs, cols=4, font_size=10)
    
    return g_img_canny


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
