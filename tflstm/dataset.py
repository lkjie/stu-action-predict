#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd

_HASH_BUCKET_SIZE = 1000

categorical_features = ['student_id', 'card_id', 'timeslot', 'trans_type', 'category', 'device_name']

continus_features = ['amount', 'remained_amount']

df = pd.read_csv('data/consum_cleaned.csv')
df[categorical_features] = df[categorical_features].fillna('')
df[continus_features] = df[continus_features].fillna(0)
df['card_id'] = df['card_id'].apply(str)
_CSV_COLUMNS = df.columns.tolist()

TOTAL_NUM = df.shape[0]
FIFTEEN_PCT = int(TOTAL_NUM * 0.15)
_NUM_EXAMPLES = {
    'train': TOTAL_NUM - FIFTEEN_PCT * 2,
    'validation': FIFTEEN_PCT,
}
_ID_SIZE = df.student_id.drop_duplicates().count()
_LABEL_NUM = 104

def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    # Continuous variable columns
    cate_columns = []
    cate_embedding_columns = []
    cate_no_embedding_columns = []
    continus_columns = []
    embedding_features = []
    for col in categorical_features:
        coluniquelen = len(df[col].unique())
        if coluniquelen > _HASH_BUCKET_SIZE:
            print("col name :{}\t\tcount:{}".format(col, coluniquelen))
            embedding_features.append(col)
    for feat in categorical_features:
        f = tf.feature_column.categorical_column_with_vocabulary_list(feat, df[feat].unique())
        if feat in embedding_features:
            f1 = tf.feature_column.embedding_column(f, dimension=100)
            cate_embedding_columns.append(f1)
        else:
            f1 = tf.feature_column.indicator_column(f)
            cate_no_embedding_columns.append(f1)
        cate_columns.append(f)
    for feat in continus_features:
        f = tf.feature_column.numeric_column(feat)
        continus_columns.append(f)
    deep_columns = cate_embedding_columns + cate_no_embedding_columns + continus_columns
    return deep_columns


def input_fn(data_file, num_epochs, shuffle, batch_size, is_classification=False):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
        '%s not found. Please make sure you have run census_dataset.py and '
        'set the --data_dir argument to the correct path.' % data_file)

    def parse_csv(value):
        tf.logging.info('Parsing {}'.format(data_file))
        record_defaults = [[0.0] if c in continus_features else [''] for c in _CSV_COLUMNS]
        columns = tf.decode_csv(value, record_defaults)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('placei')
        if is_classification:
            classes = tf.strings.to_number(labels, out_type=tf.int64)
            return features, classes
        else:
            return features, labels

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    # dataset = dataset.batch(batch_size)
    return dataset
