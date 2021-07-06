import os
import csv
import numpy as np

from datetime import datetime
from random import shuffle
from skimage import io
from sklearn.svm import LinearSVC, SVC
from shutil import copyfile

from core import create_folders, remove_files
from features import describe_color_moments, describe_haralick, describe_lbp, LBP, HARALICK, COLOR_MOMENTS


HEALTHY = 'healthy'
GLAUCOMA = 'glaucoma'


def prepare_dataset(src_path, dest_path, train_factor=0.5):
    # divide images on 70/30 for training/test
    healthy = []
    glaucoma = []

    # separate files
    with os.scandir(src_path) as entries:
        for entry in entries:
            if entry.is_file() and not entry.name.startswith('.'):
                if entry.name[0] == 'N':
                    healthy.append(entry.name)
                else:
                    glaucoma.append(entry.name)
    
    # shuffle lists
    shuffle(healthy)
    shuffle(glaucoma)

    # copy files
    healthy_limit = int(len(healthy) * train_factor)
    glaucoma_limit = int(len(glaucoma) * train_factor)

    for index, file in enumerate(healthy):
        if index > healthy_limit:
            # testing
            copyfile(src_path + file, dest_path + 'testing/' + file)
        else:
            # training
            copyfile(src_path + file, dest_path + 'training/' + file)
    
    for index, file in enumerate(glaucoma):
        if index > glaucoma_limit:
            # testing
            copyfile(src_path + file, dest_path + 'testing/' + file)
        else:
            # training
            copyfile(src_path + file, dest_path + 'training/' + file)


def analyze_features(src_path, dest_path, features):
    data = []
    labels = []

    # create paths if not exist
    dirs_to_check = [dest_path, dest_path + 'training/', dest_path + 'testing/']
    create_folders(dirs_to_check)
    
    # clean destination paths
    dirs_to_check.pop(0)
    for folder in dirs_to_check:
        remove_files(folder)
    
    # prepare dataset
    prepare_dataset(src_path, dest_path)

    # training
    print('-- TRAINING')
    with os.scandir(dest_path + 'training/') as entries:
        for entry in entries:
            if entry.is_file() and not entry.name.startswith('.'):
                print(f'Extracting features of {entry.name}')
                # read images
                img = io.imread(dest_path + 'training/' + entry.name)
                # extract features
                f_data = []
                if LBP in features:
                    f_data += describe_lbp(img)
                if HARALICK in features:
                    f_data += describe_haralick(img)
                if COLOR_MOMENTS in features:
                    f_data += describe_color_moments(img)
                
                # add data
                data.append(f_data)

                # add label
                label = HEALTHY if entry.name[0] == 'N' else GLAUCOMA
                labels.append(label)

    # train a SVM on the data
    # model = LinearSVC(C=100.0, random_state=42)
    # model = LinearSVC(random_state = 0, max_iter=1000)
    # model = LinearSVC()
    weight = {HEALTHY: 0.555, GLAUCOMA: 0.845}
    model = SVC(kernel='linear', C = 5000.0, class_weight=weight)
    model.fit(data, labels)

    # testing
    print('-- TESTING')
    csv_rows = []
    with os.scandir(dest_path + 'testing/') as entries:
        for entry in entries:
            if entry.is_file() and not entry.name.startswith('.'):
                print(f'Extracting features of {entry.name}')
                # read images
                img = io.imread(dest_path + 'testing/' + entry.name)
                # extract features
                f_data = []
                tested_features = ''
                if LBP in features:
                    f_data += describe_lbp(img)
                    tested_features += LBP + '+'
                if HARALICK in features:
                    f_data += describe_haralick(img)
                    tested_features += HARALICK + '+'
                if COLOR_MOMENTS in features:
                    f_data += describe_color_moments(img)
                    tested_features += COLOR_MOMENTS + '+'

                # predict result
                prediction = model.predict(np.array(f_data).reshape(1, -1))
                result = prediction[0]
                expected = HEALTHY if entry.name[0] == 'N' else GLAUCOMA
                msg = f'Result: {result}'
                print(msg)

                # append result on csv
                csv_rows.append([entry.name, tested_features[:-1], expected, result, expected == result])

    now = datetime.now().strftime("%Y%m%d%H%M%S")
    dest = dest_path + f'features_results_{now}.csv'
    with open(dest, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['file_name', 'features', 'expected', 'result', 'success'])
        csv_writer.writerows(csv_rows)
