#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def load_train_data(path):
    '''load raw train data'''
    raw_data = pd.read_csv(path, dtype='int16')
    raw_train_data = np.array(raw_data.values[..., 1:], dtype='int16')
    raw_train_label_list = np.array(raw_data.values[..., 0], dtype='int8')

    return raw_train_data, raw_train_label_list


def load_test_data(path):
    '''load test data'''
    raw_data = pd.read_csv(path, dtype='int16')
    test_data = np.array(raw_data.values, dtype='int16')

    return test_data


def list2one_hot(label_list):
    '''convert the 1D ndarray to one hot form'''
    label_one_hot = np.zeros([label_list.shape[0], 10], dtype='int8')
    for i in range(10):
        label_one_hot[label_list[:] == i, i] = 1

    return label_one_hot


def one_hot2list(label_one_hot):
    '''convert one hot form to 1D nd array'''
    label_list = np.where(label_one_hot[..., :] == np.amax(label_one_hot, 1)[:, np.newaxis])[1]

    return label_list


def train_data_split(raw_train_data, raw_train_label_list):
    '''split train data into two parts: train data and validation data'''
    size_data = raw_train_label_list.shape[0]
    validation_index = round(0.2 * size_data)

    index_data = np.array(range(size_data), dtype='int32')
    np.random.shuffle(index_data)

    train_data = np.array(raw_train_data[index_data[validation_index:], :], dtype='int16')
    validation_data = np.array(raw_train_data[index_data[:validation_index], :], dtype='int16')
    train_label_list = np.array(raw_train_label_list[index_data[validation_index:]], dtype='int16')
    validation_label_list = np.array(raw_train_label_list[index_data[:validation_index]], dtype='int16')

    return train_data, train_label_list, validation_data, validation_label_list
