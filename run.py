from os.path import abspath, dirname

from core import pre_process_imgs


BASE_DIR = dirname(abspath(__file__))

src_path = BASE_DIR + '/images/testing/'
dst_path = BASE_DIR + '/images/testing/processed/'

pre_process_imgs(src_path, dst_path)
