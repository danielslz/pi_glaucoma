from os.path import abspath, dirname
from skimage import io

from core import pre_process_imgs, show_imgs
from core.segmentation import draw_ellipse_fitting
from extraction.cup import cup_extraction, bulk_cup_extraction
from extraction.disc import disc_extraction, bulk_disc_extraction
from measure import get_cdr


BASE_DIR = dirname(abspath(__file__))

# ## preprocess
# src_path = BASE_DIR + '/images/testing/'
# dst_path = BASE_DIR + '/images/testing/preprocessed/'
# pre_process_imgs(src_path, dst_path)

# ## bulk extraction
# src_path = BASE_DIR + '/images/testing/preprocessed/'
# # cup
# dst_path = BASE_DIR + '/images/testing/cup/'
# bulk_cup_extraction(src_path, dst_path)
# # disc
# dst_path = BASE_DIR + '/images/testing/disc/'
# bulk_disc_extraction(src_path, dst_path)


## stand alone
src_path = BASE_DIR + '/images/testing/preprocessed/'
# open image
# img = io.imread(src_path + '/G-1-L.jpg')
# img = io.imread(src_path + '/N-7-L.jpg')
# img = io.imread(src_path + '/N-2-R.jpg')
# img = io.imread(src_path + '/N-1-L.jpg')
img = io.imread(src_path + '/N-91-L.jpg')

# cup extraction
cup_area = cup_extraction(img, show_steps=True)
img_cup = draw_ellipse_fitting(img, cup_area)

# disc extraction
disc_area = disc_extraction(img, show_steps=True)
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
