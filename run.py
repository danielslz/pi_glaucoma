from os.path import abspath, dirname
from skimage import io

from core import pre_process_imgs, show_imgs
from core.segmentation import draw_ellipse_fitting
from extraction import analyze_cdr
from extraction.cup import cup_extraction
from extraction.disc import disc_extraction
from measure import get_cdr
from features import LBP, HARALICK, COLOR_MOMENTS, describe_lbp
from features.regnize import analyze_features


BASE_DIR = dirname(abspath(__file__))

## preprocess
src_path = BASE_DIR + '/images/raw/'
dst_path = BASE_DIR + '/images/preprocessed/'
pre_process_imgs(src_path, dst_path)

## cdr extraction
src_path = BASE_DIR + '/images/preprocessed/'
dst_path = BASE_DIR + '/images/cdr/'
analyze_cdr(src_path, dst_path)

## features
src_path = BASE_DIR + '/images/preprocessed/'
dest_path = BASE_DIR + '/images/features/'

features = [LBP]
analyze_features(src_path, dest_path, features)

features = [HARALICK]
analyze_features(src_path, dest_path, features)

features = [COLOR_MOMENTS]
analyze_features(src_path, dest_path, features)

features = [LBP, HARALICK]
analyze_features(src_path, dest_path, features)

features = [LBP, COLOR_MOMENTS]
analyze_features(src_path, dest_path, features)

features = [HARALICK, COLOR_MOMENTS]
analyze_features(src_path, dest_path, features)

features = [LBP, HARALICK, COLOR_MOMENTS]
analyze_features(src_path, dest_path, features)


# # stand alone cdr extraction
# src_path = BASE_DIR + '/images/preprocessed/'

# img = io.imread(src_path + '/G-1-L.png')
# img = io.imread(src_path + '/N-7-L.png')
# img = io.imread(src_path + '/N-2-R.png')
# img = io.imread(src_path + '/N-1-L.png')


# img = io.imread(src_path + '/G-38-L.png')

# print(describe_lbp(img))

# cup extraction
# cup_area = cup_extraction(img, show_steps=True)
# img_cup = draw_ellipse_fitting(img, cup_area)

# # # disc extraction
# disc_area = disc_extraction(img, show_steps=True)
# img_disc = draw_ellipse_fitting(img, disc_area)

# # # disc and cup on original
# img_final = draw_ellipse_fitting(img, cup_area)
# img_final = draw_ellipse_fitting(img_final, disc_area)

# # # get cdr
# cdr = get_cdr(cup_area, disc_area)
# result = 'normal' if cdr <= 0.5 else 'glaucoma'

# # show images
# imgs = [
#     {'data': img, 'title': 'original'},
#     {'data': img_cup, 'title': 'cup extraction'},
#     {'data': img_disc, 'title': 'disc extraction'},
#     {'data': img_final, 'title': f'result: {result}, cdr: {cdr:0.3f}'},
# ]

# show_imgs(imgs)
