import os
import csv

from skimage import io
from measure import get_cdr

from core import add_text, remove_files, create_folders
from core.segmentation import draw_ellipse_fitting
from extraction.cup import cup_extraction
from extraction.disc import disc_extraction


HEALTHY = 'healthy'
GLAUCOMA = 'glaucoma'

def analyze_cdr(src_path, dest_path):
    # create paths if not exist
    dirs_to_check = [dest_path, dest_path + HEALTHY + '/', dest_path + GLAUCOMA + '/']
    create_folders(dirs_to_check)
    
    # clean destination paths
    for folder in dirs_to_check:
        remove_files(folder)
    
    csv_rows = []
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
                expected = HEALTHY if entry.name[0] == 'N' else GLAUCOMA
                msg = f'Result: {result}, CDR: {cdr:0.3f}'
                print(msg)
                # mark cup and disc on original
                new_img = draw_ellipse_fitting(original, cup)
                new_img = draw_ellipse_fitting(new_img, disc)
                new_img = add_text(new_img, msg)
                # save on dest
                result_path = result + '/'
                io.imsave(dest_path + result_path + entry.name, new_img)
                # append result on csv
                csv_rows.append([entry.name, cdr, expected, result, expected == result])
                # remove preprocessed file
                # os.remove(src_path + entry.name)

    dest = dest_path + "cdr_results.csv"
    with open(dest, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['file_name', 'cdr', 'expected', 'result', 'success'])
        csv_writer.writerows(csv_rows)
