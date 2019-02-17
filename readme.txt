本项目是Kaggle竞赛平台的手写数字识别项目，数据可在平台上下载。
链接：https://www.kaggle.com/c/digit-recognizer
1、本项目作为深度神经网络学习入门训练，提供两种解决方案，CNN和MLP。
2、在当前目录建立data文件夹，并将下载的数据文件放在其中。还需在当前目录建立logs（存放tensorboard的神经网络可视化图），fit_trend（存放训练过程图）两个文件夹。
3、myLoadData.py是训练和测试数据导入模块，里面还有1D矢量和onehot形式转化函数，训练集拆分的函数。
4、CNN_keras.py是CNN网络实现模块。MLP_keras.py是MLP实现模块。
5、DataVisualize.py是训练集和测试集plot模块，用于前期分析调试程序。
6、CNN调用方式，在cmd输入：python CNN_keras.py；MLP调用方式，在cmd输入：python MLP_keras.py
