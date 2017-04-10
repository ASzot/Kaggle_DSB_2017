import os
import numpy as np
import pandas as pd
from model.resnet50 import ResNet50
import preprocessing as prep
import extractor


if __name__ == '__main__':
    print('Creating resnet')
    model = extractor.get_extractor()
    print('Finished creating resnet')

    data_dir = '/hdd/datasciencebowl2017/stage1/'
    patients = os.listdir(data_dir)
    labels_df = pd.read_csv('./data/stage1_labels.csv', index_col = 0)

    for patient in patients[:10]:
        try:
            label = labels_df.get_value(patient, 'cancer')
        except:
            print('Not labeled')

        path = data_dir + patient
        scan = prep.load_scan(path)
        if scan is None:
            print('Could not load scan')
            continue

        scan = prep.full_preprocess(scan)
        scan = scan.reshape(-1, 224, 224, 3)
        predictions = model.predict(scan, verbose=1)


