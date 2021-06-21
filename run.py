from os.path import abspath, dirname
from skimage import io

from core import pre_process_imgs, show_imgs
from extraction.cup import cup_extraction, bulk_cup_extraction


BASE_DIR = dirname(abspath(__file__))

# src_path = BASE_DIR + '/images/testing/'
dst_path = BASE_DIR + '/images/testing/pre-processed/'
# pre_process_imgs(src_path, dst_path)

# src_path = BASE_DIR + '/images/testing/pre-processed/'
# dst_path = BASE_DIR + '/images/testing/cup/'
# bulk_cup_extraction(src_path, dst_path)


# open image
img = io.imread(dst_path + '/N-7-L.jpg')
# img = io.imread(dst_path + '/N-2-R.jpg')
# img = io.imread(dst_path + '/N-1-L.jpg')

# cup extraction
img_cup = cup_extraction(img, True)

# show images
imgs = [
    {'data': img, 'title': 'original'},
    {'data': img_cup, 'title': 'cup extraction'},
]

show_imgs(imgs)
