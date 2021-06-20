import numpy as np

from os.path import abspath, dirname
from skimage import io, exposure
from skimage.feature import canny
from skimage.util import invert
from skimage.morphology import disk, opening, convex_hull_image

from core import pre_process_imgs, show_imgs
from core.segmentation import region_growing


BASE_DIR = dirname(abspath(__file__))

src_path = BASE_DIR + '/images/testing/'
dst_path = BASE_DIR + '/images/testing/processed/'

# pre_process_imgs(src_path, dst_path)

# open image
img = io.imread(dst_path + '/N-7-L.jpg')


######  cup detection

# separate channels
r_img = img[:, :, 0]
g_img = img[:, :, 1]
b_img = img[:, :, 2]

# contrast stretching
p2, p98 = np.percentile(img, (2, 98))
g_img_rescale = exposure.rescale_intensity(g_img, in_range=(p2, p98))

# negative transform
g_img_inverted = invert(g_img_rescale)

# opening
g_img_opening = opening(g_img_inverted, disk(25))  # disk(40)

# negative transform
g_img_inverted_2 = invert(g_img_opening)

# region grow
seed = np.unravel_index(g_img_inverted_2.argmax(), g_img_inverted_2.shape)
g_img_region_grow = region_growing(g_img_inverted_2, seed, threshold=50)

# convex hull
g_img_convex_hull = convex_hull_image(g_img_region_grow)

# canny edge detector
g_img_canny = canny(g_img_convex_hull)

# show images
imgs = [
    {'data': img, 'title': 'original'},
    {'data': g_img, 'title': 'green channel'},
    {'data': g_img_rescale, 'title': 'contrast stretching'},
    {'data': g_img_inverted, 'title': 'negative transform'},
    {'data': g_img_opening, 'title': 'opening'},
    {'data': g_img_inverted_2, 'title': 'negative transform'},
    {'data': g_img_region_grow, 'title': 'region grow'},
    {'data': g_img_convex_hull, 'title': 'convex hull'},
    {'data': g_img_canny, 'title': 'canny edge detector'},
]

show_imgs(imgs)
