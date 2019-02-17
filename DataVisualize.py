import matplotlib.pyplot as plt
import myLoadData
import numpy as np
import os


def train_data_view():
    trainData, trainLabels = myLoadData.load_train_data(os.path.join('data', 'train.csv'))
    size = len(trainLabels)  # 总的样本点个数
    indexArray = np.array(range(size), dtype=int)  # 下标数组
    np.random.shuffle(indexArray)
    fig = plt.figure()
    for i in range(4):
        plt.subplot(221 + i)
        image = trainData[indexArray[i], :]
        image = np.reshape(image, [28, 28])
        plt.imshow(image, cmap='gray')
        plt.title(trainLabels[indexArray[i]])
    fig.savefig(r'train_digits.png')
    print(trainLabels[indexArray[0:4]])


def test_data_view():
    test_data = myLoadData.load_test_data(os.path.join('data', 'test.csv'))
    fig = plt.figure()
    for i in range(16):
        plt.subplot(4, 4, 1 + i)
        image = test_data[i, :]
        image = np.reshape(image, [28, 28])
        plt.imshow(image, cmap='gray')

    fig.savefig(r'test_digits.png')


if __name__ == '__main__':
    # 导入输入训练集
    test_data_view()
