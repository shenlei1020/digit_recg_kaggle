#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import time
import myLoadData

if __name__ == '__main__':
    start = time.clock()

    # Load data
    raw_train_data, raw_train_label_list = myLoadData.load_train_data(os.path.join('data', 'train.csv'))  # train data
    # generate train data and validation data
    train_data, train_label_list, validation_data, validation_label_list = \
        myLoadData.train_data_split(raw_train_data, raw_train_label_list)

    train_label = myLoadData.list2one_hot(train_label_list)  # convert to one hot form
    validation_label = myLoadData.list2one_hot(validation_label_list)  # convert to one hot form

    test_data = myLoadData.load_test_data(os.path.join('data', 'test.csv'))  # test data

    # convert data type
    train_data = train_data.astype(np.float32)
    validation_data = validation_data.astype(np.float32)
    train_label = train_label.astype(np.float32)
    validation_label = validation_label.astype(np.float32)

    test_data = test_data.astype(np.float32)

    # Convert the pixel values from integers between 0 and 255 to floats between 0 and 1
    train_data /= 255
    validation_data /= 255
    test_data /= 255

    tf.keras.backend.clear_session()

    if not ('MLP_keras.h5' in os.listdir()):

        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Dense(300, activation=tf.nn.relu, bias_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(784,), name='dense_1'))  # [None, 780]
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(120, activation=tf.nn.relu, bias_regularizer=tf.keras.regularizers.l2(0.01), name='dense_2'))  # [None, 150]
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='dense_3'))  # <- [None, 10]
        # 注：每个层都自定义名称，用于tensorboard观察

        optimizer = tf.keras.optimizers.Adam(lr=0.001)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        model.summary()

        BATCH_SIZE = 128
        EPOCHS = 30

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.000015*0.2),
                     tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0)
                     ]
        fit_history = model.fit(train_data, train_label, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                validation_data=(validation_data, validation_label),
                                callbacks=callbacks)

        fig_acc = plt.figure()
        plt.plot(fit_history.history['acc'])
        plt.plot(fit_history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower right')
        fig_acc.savefig(r'fit_trend\MLP_keras_acc.png')
        fig_loss = plt.figure()
        plt.plot(fit_history.history['loss'])
        plt.plot(fit_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        fig_loss.savefig(r'fit_trend\MLP_keras_loss.png')

    else:
        model = tf.keras.models.load_model('MLP_keras.h5')
        model.summary()

    print("Go to Prediction ... ")
    test_label = model.predict(test_data, batch_size=128)
    test_label_list = myLoadData.one_hot2list(test_label)
    print("Test label list:\n", test_label_list)

    model.save('MLP_keras_250_100_valiratio_0.1.h5')

    # 保存我的submission
    test_ImageId = np.array(range(len(test_label_list)), dtype='int32')[:, np.newaxis] + 1
    test_submit_values = np.concatenate((test_ImageId, test_label_list[:, np.newaxis]), axis=1)
    test_submit = pd.DataFrame(test_submit_values, columns=['ImageId', 'Label'])
    test_submit.to_csv(r"data\MLP_keras_submit.csv", index=False)

    print("Time used:", (time.clock() - start))
