from os.path import abspath, dirname
from skimage import io

from core import pre_process_imgs, show_imgs
from core.segmentation import draw_ellipse_fitting
from extraction.cup import cup_extraction, bulk_cup_extraction
from extraction.disc import disc_extraction
from measure import get_cdr


BASE_DIR = dirname(abspath(__file__))

# src_path = BASE_DIR + '/images/testing/'
dst_path = BASE_DIR + '/images/testing/pre-processed/'
# pre_process_imgs(src_path, dst_path)

# open image
img = io.imread(dst_path + '/G-1-L.jpg')
# img = io.imread(dst_path + '/N-7-L.jpg')
# img = io.imread(dst_path + '/N-2-R.jpg')
# img = io.imread(dst_path + '/N-1-L.jpg')

# cup extraction
cup_area = cup_extraction(img, return_cup_area=True)
img_cup = draw_ellipse_fitting(img, cup_area)

# src_path = BASE_DIR + '/images/testing/pre-processed/'
# dst_path = BASE_DIR + '/images/testing/cup/'
# bulk_cup_extraction(src_path, dst_path)

# disc extraction
disc_area = disc_extraction(img, return_disc_area=True)
img_disc = draw_ellipse_fitting(img, disc_area)

# disc and cup on original
img_final = draw_ellipse_fitting(img, cup_area)
img_final = draw_ellipse_fitting(img_final, disc_area)

# get cdr
cdr = get_cdr(cup_area, disc_area)
result = 'normal' if cdr <= 0.5 else 'glaucoma'

# show images
imgs = [
    {'data': img, 'title': 'original'},
    {'data': img_cup, 'title': 'cup extraction'},
    {'data': img_disc, 'title': 'disc extraction'},
    {'data': img_final, 'title': f'result: {result}, cdr: {cdr:0.3f}'},
]

show_imgs(imgs)
