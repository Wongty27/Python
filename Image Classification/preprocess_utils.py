import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.datasets import make_imbalance
import tensorflow as tf
from tensorflow import keras

def resize_rescale(image, label):
    image =  tf.clip_by_value(image / 255.0, 0.0, 1.0)
    return image, label

def get_dataset(root_dir, image_size):
    ds = keras.utils.image_dataset_from_directory(
        directory=root_dir, labels='inferred', label_mode='int', shuffle=True,
        color_mode='rgb', crop_to_aspect_ratio=True, batch_size=None,
        image_size=(image_size, image_size), interpolation='bicubic')
    class_names = ds.class_names
    ds = ds.map(resize_rescale)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # feature_list = [x for x, _ in ds.map(lambda x, y: x,)]
    # label_list = [y for _, y in ds.map(lambda x, y: y)]

    feature_list = []
    label_list = []
    for x, y in ds.map(lambda x, y: (x, y)):
        feature_list.append(x)
        label_list.append(y)
    x_true = np.asarray(feature_list)
    batch_size, height, width, channel = x_true.shape
    x_flat = x_true.reshape((batch_size, height*width*channel))
    y_true = np.asarray(label_list)
    y_flat = y_true.flatten()
    print(Counter(y_flat)) # show sample size for each class
    return x_true, y_flat, class_names

def sample_per_class(features, labels, n_samples, augment=False):
    '''Balance the sample size of each dataset class.'''
    batch_size, height, width, channel = features.shape
    features_reshaped = features.reshape((-1, height*width*channel))
    if augment:
        sampling_strat = {0: n_samples, 1: n_samples, 2: 0, 3: n_samples}
    else:
        sampling_strat = {0: n_samples, 1: n_samples, 2: n_samples, 3: n_samples}
    x_resampled, y_resampled = make_imbalance(
        features_reshaped, labels, sampling_strategy=sampling_strat, random_state=7, verbose=True)
    x_resampled = x_resampled.reshape((len(x_resampled), height, width, channel))
    return x_resampled, y_resampled

def train_split(features, labels, train_ratio):
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=7)
    for i, (train_idx, test_idx) in enumerate(splitter.split(features, labels)):
        x_train = [features[i] for i in train_idx]
        y_train = [labels[i] for i in train_idx]
        x_test = [features[i] for i in test_idx]
        y_test = [labels[i] for i in test_idx]
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    return x_train, y_train, x_test, y_test

def val_test_split(features, labels, val_ratio):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=7)
    for i, (val_idx, test_idx) in enumerate(splitter.split(features, labels)):
        x_val = [features[i] for i in val_idx]
        y_val = [labels[i] for i in val_idx]
        x_test = [features[i] for i in test_idx]
        y_test = [labels[i] for i in test_idx]
    x_val = np.asarray(x_val)
    y_val = np.asarray(y_val)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    return x_val, y_val, x_test, y_test

def imbalance_train(features, labels, minor_ratio):
    batch_size, height, width, channel = features.shape
    features_reshaped = features.reshape((-1, height*width*channel))
    count = Counter(labels)
    n_major = count[2] # change the number according to intended majority class 
    n_minor = int(n_major * minor_ratio)
    sampling_strat = {0: n_minor, 1: n_minor, 2: n_major, 3: n_minor}
    x_resampled, y_resampled = make_imbalance(
        features_reshaped, labels, sampling_strategy=sampling_strat, random_state=7, verbose=True)
    x_resampled = x_resampled.reshape((len(x_resampled), height, width, channel))
    return x_resampled,y_resampled