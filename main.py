import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model.resnet50 import ResNet50
import preprocessing as prep
import extractor
import xgboost as xgb
from sklearn import cross_validation, metrics


def create_features(patients):
    print('Creating resnet')
    model = extractor.get_extractor()
    print('Finished creating resnet')

    count_index = 0
    for patient in patients:
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
        features = model.predict(scan, verbose=1)

        np.save('/hdd/datasciencebowl2017/stage1features/' + patient, features)

        count_index += 1
        print('%.2f' % ((float(count_index) / float(len(patients))) * 100.))


def train_decision(labels_df):
    root_folder = '/hdd/datasciencebowl2017/stage1features/'
    labels = labels_df['id'].tolist()

    x = [np.mean(np.load(root_folder + '%s.npy' % (patient_id)), axis=0) for patient_id
            in labels]
    y = labels_df['cancer'].tolist()

    # Append the other samples.
    #s1_answers_df = pd.read_csv('/hdd/datasciencebowl2017/stage1_solution.csv')
    #x.extend([np.mean(np.load(root_folder + '%s.npy' % (patient_id)), axis=0)
    #    for patient_id in s1_answers_df['id'].tolist()])

    #y.extend(s1_answers_df['cancer'].tolist())

    xy = list(zip(*[(x_sample, y_sample) for x_sample, y_sample in zip(x,y) if
            not np.isnan(x_sample).any()]))

    x = np.array(xy[0])
    y = np.array(xy[1])

    xgbr = xgb.XGBRegressor(max_depth=20,
                           n_estimators=10000,
                           min_child_weight=20,
                           learning_rate=0.05,
                           subsample=0.80,
                           colsample_bytree=0.80,
                           seed=4242)

    x = x.reshape(-1, 2048)

    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42,
            stratify=y, test_size=0.20)

    print(trn_x.shape)
    print(trn_y.shape)
    print(val_x.shape)
    print(val_y.shape)

    xgbr.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True,
            eval_metric='logloss', early_stopping_rounds=50)

    return xgbr


def make_predictions(xgbr):
    df = pd.read_csv('/hdd/datasciencebowl2017/stage2_sample_submission.csv')
    x = np.array([np.mean(np.load('/hdd/datasciencebowl2017/stage1features/%s.npy' % str(patient_id)), axis=0)
        for patient_id in df['id'].tolist()])

    x = x.reshape(-1, 2048)

    pred_y = xgbr.predict(x)

    df['cancer'] = pred_y

    df.to_csv('submission.csv', index=False)


if __name__ == '__main__':

    data_dir = '/hdd/datasciencebowl2017/stage2/'
    patients = os.listdir(data_dir)
    labels_df = pd.read_csv('./data/stage1_labels.csv')

    count_index = 0;

    #create_features(patients)
    xgbr = train_decision(labels_df)
    make_predictions(xgbr)

