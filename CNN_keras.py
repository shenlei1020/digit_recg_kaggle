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

    train_data = train_data.astype(np.float32)
    validation_data = validation_data.astype(np.float32)
    train_label = train_label.astype(np.float32)
    validation_label = validation_label.astype(np.float32)

    test_data = test_data.astype(np.float32)

    # Convert the grey values from 0 ~ 255 to 0 ~ 1
    train_data /= 255
    validation_data /= 255
    test_data /= 255

    tf.keras.backend.clear_session()

    if not ('CNN_keras.h5' in os.listdir()):

        model = tf.keras.Sequential()
        # with some simplifications:
        #   (1) The first Conv2D uses default initialization.
        #   (2) Use max pooling for subsampling and remove upsampling.
        #   (3) Use ReLU as activation functions.

        model.add(tf.keras.layers.Reshape((28, 28, 1), input_shape=(784,), name='reshape'))   # [None, 28, 28, 1]
        model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(5, 5), padding='same', activation=tf.nn.relu, name='conv_1'))  # [None, 28, 28, 6]
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='maxpool_1'))  # <- [None, 14, 14, 6]
        model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation=tf.nn.relu, name='conv_2'))  # [None, 10, 10, 16]
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='maxpool_2'))  # <- [None, 5, 5, 16]
        model.add(tf.keras.layers.Flatten(name='flatten'))   # <- [None, 400]
        model.add(tf.keras.layers.Dense(120, activation=tf.nn.relu, bias_regularizer=tf.keras.regularizers.l2(0.01), name='dense_1'))  # [None, 120]
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(84, activation=tf.nn.relu, bias_regularizer=tf.keras.regularizers.l2(0.01), name='dense_2'))  # [None, 84]
        # model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='dense_3'))  # <- [None, 10]
        # 注：每个层都自定义名称，方便tensorboard观察

        optimizer = tf.keras.optimizers.Adam(lr=0.002)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        model.summary()

        BATCH_SIZE = 128  # 128
        EPOCHS = 30

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.000015 * 0.2),  # 0.000015
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
        fig_acc.savefig(r'fit_trend\CNN_keras_acc.png')
        fig_loss = plt.figure()
        plt.plot(fit_history.history['loss'])
        plt.plot(fit_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        fig_loss.savefig(r'fit_trend\CNN_keras_loss.png')

    else:
        model = tf.keras.models.load_model('CNN_keras.h5')
        model.summary()

    print("Go to Prediction ... ")
    test_label = model.predict(test_data, batch_size=128)
    test_label_list = myLoadData.one_hot2list(test_label)
    print("Test label list:\n", test_label_list)

    model.save('CNN_keras_6_16_valiratio_0.1.h5')

    # 保存我的submission
    test_ImageId = np.array(range(len(test_label_list)), dtype='int32')[:, np.newaxis] + 1
    test_submit_values = np.concatenate((test_ImageId, test_label_list[:, np.newaxis]), axis=1)
    test_submit = pd.DataFrame(test_submit_values, columns=['ImageId', 'Label'])
    test_submit.to_csv(r"data\CNN_keras_submit.csv", index=False)

    print("Time used:", (time.clock() - start))
