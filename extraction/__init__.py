import os

from skimage import io
from measure import get_cdr

from core import add_text
from core.segmentation import draw_ellipse_fitting
from extraction.cup import cup_extraction
from extraction.disc import disc_extraction


HEALTHY = 'healthy'
GLAUCOMA = 'glaucoma'

def analyze_cdr(src_path, dest_path):
    dirs_to_check = [dest_path, dest_path + HEALTHY + '/', dest_path + GLAUCOMA + '/']
    # create dest if not exist
    for dir in dirs_to_check:
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    with os.scandir(src_path) as entries:
        for entry in entries:
            if entry.is_file() and not entry.name.startswith('.'):
                print(f'Analyzing CDR of {entry.name}')
                # read images
                original = io.imread(src_path + entry.name)
                cup = cup_extraction(original)
                disc = disc_extraction(original)
                # compute cdr
                cdr = get_cdr(cup, disc)
                result = HEALTHY if cdr <= 0.5 else GLAUCOMA
                msg = f'Result: {result}, CDR: {cdr:0.3f}'
                print(msg)
                # mark cup and disc on original
                new_img = draw_ellipse_fitting(original, cup)
                new_img = draw_ellipse_fitting(new_img, disc)
                new_img = add_text(new_img, msg)
                # save on dest
                result_path = result + '/'
                io.imsave(dest_path + result_path + entry.name, new_img)
