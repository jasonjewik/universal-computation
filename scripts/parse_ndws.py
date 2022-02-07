from typing import Dict, List, Text, Tuple

import numpy as np
import tensorflow as tf

"""
This script converts the Next Day Wildfire Spread (NDWS) dataset from its
TFRecord format to NPZ format. It assumes the TFRecord files are stored at
data/next-day-wildfire-spread. E.g.,
    
    data/next-day-wildfire-spread/next_day_wildfire_spread_train00.tfrecord
    data/next-day-wildfire-spread/next_day_wildfire_spread_eval00.tfrecord
    data/next-day-wildfire-spread/next_day_wildfire_spread_test00.tfrecord

The train files are stored in data/ndws_train.npz. From there, they can be
loaded like this

    train_dict = np.load('data/ndws/ndws_train.npz')
    train_dict[train_partition]

where train_partition is train_00, train_01, ..., train_14. The evaluation and
test NPZ files follow a similar pattern, except replace 'eval' with 'val' for
the evaluation data.

Data shape: (batch, height, width, channels).
There are 13 channels for each data example. These channels are:
    - elevation: in meters
    - th: wind direction in degrees clockwise from north
    - vs: wind speed in m/s
    - tmmn: min temperature in Kelvin
    - tmmx: max temperature in Kelvin
    - sph: specific humidity
    - pr: precipitation in mm
    - pdsi: pressure
    - NDVI: normalized difference vegetation index
    - population: population density
    - erc: NFDRS fire danger index energy release component (BTU/ft^2)
    - PrevFireMask: 1 = fire, 0 = no fire, -1 = unlabeled
    - FireMask: target, same labels as PrevFireMask

The original NDWS dataset can be accessed here:
https://www.kaggle.com/fantineh/next-day-wildfire-spread
"""


def main():
    data_size = 64   # km^2
    INPUT_FEATURES = ['elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph',
                      'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']
    OUTPUT_FEATURES = ['FireMask']
    direc = 'data/next-day-wildfire-spread'

    train_data = dict()
    for i in range(15):
        num = f'{i:02}'
        fp = f'{direc}/next_day_wildfire_spread_train_{num}.tfrecord'
        arr = get_np_arr(fp, data_size, INPUT_FEATURES, OUTPUT_FEATURES)
        train_data[f'train_{num}'] = arr
    np.savez_compressed('data/ndws/ndws_train', **train_data)

    val_data, test_data = dict(), dict()
    for i in range(2):
        num = f'{i:02}'

        fp = f'{direc}/next_day_wildfire_spread_eval_{num}.tfrecord'
        arr = get_np_arr(fp, data_size, INPUT_FEATURES, OUTPUT_FEATURES)
        val_data[f'val_{num}'] = arr

        fp = f'{direc}/next_day_wildfire_spread_test_{num}.tfrecord'
        arr = get_np_arr(fp, data_size, INPUT_FEATURES, OUTPUT_FEATURES)
        test_data[f'test_{num}'] = arr

    np.savez_compressed('data/ndws/ndws_val.npz', **val_data)
    np.savez_compressed('data/ndws/ndws_test.npz', **test_data)


def get_features_dict(
    sample_size: int,
    features: List[Text]
) -> Dict[Text, tf.io.FixedLenFeature]:
    sample_shape = [sample_size, sample_size]
    features = set(features)
    columns = [
        tf.io.FixedLenFeature(shape=sample_shape, dtype=tf.float32)
        for _ in features
    ]
    return dict(zip(features, columns))


def parse(
    example_proto: tf.train.Example,
    data_size: int,
    input_features: List[Text],
    output_features: List[Text]
) -> Tuple[tf.Tensor, tf.Tensor]:
    feature_names = input_features + output_features
    features_dict = get_features_dict(data_size, feature_names)
    features = tf.io.parse_single_example(example_proto, features_dict)
    inputs_list = [features.get(key) for key in input_features]
    inputs_stacked = tf.stack(inputs_list, axis=0)
    input_img = tf.transpose(inputs_stacked, [1, 2, 0])
    outputs_list = [features.get(key) for key in output_features]
    outputs_stacked = tf.stack(outputs_list, axis=0)
    output_img = tf.transpose(outputs_stacked, [1, 2, 0])
    return input_img, output_img


def get_dataset(
    file_pattern: Text,
    data_size: int,
    input_features: List[Text],
    output_features: List[Text]
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.list_files(file_pattern)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type=None)
    )
    dataset = dataset.map(
        lambda x: parse(x, data_size, input_features, output_features)
    )
    return dataset


def get_np_arr(
    input_file_pattern: Text,
    data_size: int,
    input_features: List[Text],
    output_features: List[Text]
) -> np.ndarray:
    dataset = get_dataset(
        input_file_pattern,
        data_size,
        input_features,
        output_features
    )
    arr = []
    for x, y in dataset.as_numpy_iterator():
        arr.append(np.concatenate((x, y), -1))
    arr = np.array(arr)
    return arr


if __name__ == '__main__':
    main()
