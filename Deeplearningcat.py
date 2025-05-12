import numpy as np
import os
# import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.filedialog
import h5py

""" 0.数据读取与处理
    1.初始化参数
    2.前向线性传播
    3.计算激活值
    4.计算误差
    5.反向传播
    6.更新参数
    7.预测
    8.额外功能"""



def Init_params(layers):  # 初始化权重矩阵和偏置
    np.random.seed(3)  # 保证每次初始化一样
    parameters = {}  # 该字典用来储存参数
    L = len(layers)  # 神经网络的层数

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers[l],
                                                   layers[l - 1]) / np.sqrt(layers[l - 1])  # Xaiver初始化方法
        parameters['b'+str(l)] = np.zeros((layers_dims[l], 1))  # 初始化为0

    return parameters


def TanH(Z):
    return (np.exp(2*Z)-1)/(np.exp(2*Z)+1)


def Sigmoid(Z):
    return 1/(1+np.exp(-Z))


def Linear_forward(A, W, b):  # 正向线性传播
    return np.dot(W, A) + b


def Activation_forward(A_pre, W, b, Type='Hiden'):  # 计算激活值
    """
    Z表示经过线性传播后的矩阵，将输给激活函数
    A_pre表示前一层的激活值，将输给线性传播单元
    b将先广播至与W一样的大小，再进行运算
    """
    Z = Linear_forward(A_pre, W, b)
    cache = (A_pre, W, b)
    if Type == "Output":
        A = Sigmoid(Z)
    elif Type == "Hiden":
        A = TanH(Z)

    return A, cache


def Forward_propagation(X, parameters):  # 向前传播
    """
    caches用于储存cache
    每一层的激活值A将输给下一层并作用于线性传播函数
    输出层的激活值为Yhat，将输给代价函数
    """
    caches = []
    A = X
    L = len(parameters) // 2  # 获得整型
    for l in range(1, L):  # （1,3）
        A, cache = Activation_forward(
            A, parameters['W' + str(l)], parameters['b' + str(l)], "Hiden")
        caches.append(cache)
    Yhat, cache = Activation_forward(
        A, parameters['W' + str(L)], parameters['b' + str(L)], "Output")
    caches.append(cache)

    return Yhat, caches


def Compute_cost(Yhat, Y):
    m = Y.shape[1]  # 图片张数
    cost = -np.sum(np.multiply(np.log(Yhat), Y) +
                   np.multiply(np.log(1 - Yhat), 1 - Y)) / m  # 交叉熵误差计算
    # 计算Yhat的梯度，由此开始反向传播
    dYhat = - (np.divide(Y, Yhat) - np.divide(1 - Y, 1 - Yhat))
    return cost, dYhat


def Linear_backward(dZ, cache):
    A, W, b = cache  # 拆分cache
    m = A.shape[1]  # 获得图片张数
    # 除以m防止样本过大而导致数据过大
    dW = np.dot(dZ, A.T) / m  # dW/dZ=A.T,相乘代表与cost的梯度
    db = np.sum(dZ, axis=1, keepdims=True) / m  # db/dZ=I,保持维度不变
    dA = np.dot(W.T, dZ)

    return dA, dW, db


def Sigmoid_backward(dA, A):
    dZ = dA * A*(1-A)  # 相对误差的梯度
    return dZ


def TanH_backward(dA, A):
    dZ = dA*(1-A**2)  # 相对误差的梯度
    return dZ


def Activation_backward(dA, cache, A_next, activation="Hiden"):
    """
    cache储存A_pre，W，b
    A_next为输给下一层的激活值，即本层的激活值
    """
    if activation == "Hiden":
        dZ = TanH_backward(dA, A_next)
    elif activation == "Output":
        dZ = Sigmoid_backward(dA, A_next)
    dA, dW, db = Linear_backward(dZ, cache)

    return dA, dW, db


def Backward_propagation(dYhat, Yhat, Y, caches):
    grads = {}  # 用于储存梯度矩阵
    L = len(caches)  # 4
    m = Y.shape[1]  # 图片个数
    # 输出层
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)
                                                      ] = Activation_backward(dYhat, caches[L-1], Yhat, "Output")
    for l in reversed(range(L-1)):  # (3,0]
        grads["dA" + str(l + 1)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)
                                                                  ] = Activation_backward(grads["dA" + str(l + 2)], caches[l], caches[l+1][0], "Hiden")

    return grads


def Update_params(parameters, grads, learning_rate):
    # 梯度下降更新参数
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * \
            grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * \
            grads["db" + str(l + 1)]
    return parameters


def Predict(X, parameters):  # 预测
    # 将数据和训练好的参数进行一次预测
    probs, caches = Forward_propagation(X, parameters)
    if X.shape[1] == 1:  # 判断输入的是否是一张图片
        print(f'有{probs[0][0]*100}%概率是猫')
    return probs


def Accuracy(name, probs, y):  # 准确率
    p = []  # 储存预测值
    for i in range(0, probs.shape[1]):
        p.append(int(1) if probs[0, i] > 0.5 else int(0))  # 分类
    print(f"{name}准确度为: {100*np.mean((p == y))}%")

    return p


def Save_params(parameters, layers, path):
    # 储存神经网络各层的信息
    np.savetxt(path+'layers.csv', layers, delimiter=',')
    n = len(parameters)//2
    # 将每个参数分开储存，方便读取
    for i in range(1, n+1):
        np.savetxt(path+'W'+str(i)+'.csv',
                   parameters['W'+str(i)], delimiter=',')
        np.savetxt(path+'b'+str(i)+'.csv',
                   parameters['b'+str(i)], delimiter=',')


def Load_params(path):
    parameters = {}  # 用于接收参数
    layers = list(np.loadtxt(path+'layers.csv', dtype=int, delimiter=','))
    n = len(layers)
    for i in range(1, n):
        parameters['W'+str(i)] = np.loadtxt(path+'W'+str(i) +
                                            '.csv', delimiter=",").reshape(layers[i], -1)
        parameters['b'+str(i)] = np.loadtxt(path+'b'+str(i) +
                                            '.csv', delimiter=",").reshape(layers[i], 1)
    return layers, parameters

def Plot(costs, layers, learning_rate):
    """
    costs储存每100次迭代后的误差值
    layers是神经网络的信息
    """
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" +
              str(learning_rate) + f",layers={layers}")
    save_path = 'output\plot.png'  # 指定保存的路径和文件名
    plt.savefig(save_path)
    plt.show()

def Find_wrong(p, label, data):
    """
    p为预测的标签，是一个列表
    label为真实的标签，是一个1行n列的矩阵
    data为经过处理后的图片矩阵
    """
    m = len(p)
    diff = []  # 储存预测错误图片的索引值
    for i in range(m):
        if label[0, i] != p[i]:
            diff.append(i)

def Train_model(X, Y, parameters, learning_rate, iterations, threshold):  # 训练用模块
    costs = []  # 储存每100次迭代的损失值，用于绘制折线图
    for i in range(iterations):
        Yhat, caches = Forward_propagation(X, parameters)  # 正向传播
        cost, dYhat = Compute_cost(Yhat, Y)  # 计算误差
        grads = Backward_propagation(dYhat, Yhat, Y, caches)  # 计算梯度
        parameters = Update_params(
            parameters, grads, learning_rate)  # 更新参数
        if i % 100 == 0:
            costs.append(cost)
            print(f"迭代次数：{i}，误差值：{cost}")
        if cost < threshold:
            costs.append(cost)
            print(f"迭代次数：{i}，误差值：{cost}")
            break
    return parameters, costs, i


def BP(X, Y, test_data, test_label, path,  layers, iterations, learning_rate, threshold=0.06,
       find_wrong=False, save_params=False, load_params=False, continue_train=False, plot=True):
    parameters = Init_params(layers)  # 接收初始化的参数
    if load_params or continue_train:  # 继续训练也需要读取储存的参数
        layers, parameters = Load_params(path)
    # 当不继续训练或不加载数据时
    if continue_train or not load_params:
        parameters, costs, times = Train_model(
            X, Y, parameters, learning_rate, iterations, threshold)
        if save_params:
            Save_params(parameters, layers, path)
    train_probs = Predict(X, parameters)
    train_p = Accuracy('训练集', train_probs, Y)
    test_probs = Predict(test_data, parameters)
    test_p = Accuracy('测试集', test_probs, test_label)
    if find_wrong:
        Find_wrong(train_p, Y, X)
        Find_wrong(test_p, test_label, test_data)
    if plot:
        Plot(costs, layers, learning_rate)
    return parameters


def load_dataset():
    train_dataset = h5py.File('output/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('output/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def switch(s, addr, parameters, iterations, learning_rate):
    if s==1:
        # 文件的路径
        Path_params = 'E:/'

        train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
        # 测试集和训练集和图片矩阵纵向维度保持一致
        # 将训练集的维度降低并转置。
        train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
        # 将测试集的维度降低并转置。
        test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

        train_set_x = train_set_x_flatten / 255
        test_set_x = test_set_x_flatten / 255

        # 各层的节点数
        global layers_dims
        layers_dims = [train_set_x.shape[0], 20, 8, 6, 1]
        # global learning_rate
        # learning_rate = 0.0075

        # global parameters
        parameters=BP(train_set_x, train_set_y, test_set_x, test_set_y, Path_params, layers_dims,
           iterations, learning_rate, threshold=0.006, find_wrong=False, save_params=False,
           load_params=False, continue_train=False, plot=True)
        """
            train_data，train_label为训练集的图片数据和标签，test_data，test_label为测试集数据和标签
            path为保存和读取参数的绝对路径n
            layers储存了每层神经元个数
            interactions为迭代次数
            threshold为阈值，误差小于该值可结束程序

            find_wrong：是否显示识别错误的图像
            save_params：是否储存训练好的参数
            load_params：是否读取参数
            continue_train：是否读取参数继续训练
            plot：是否显示误差与迭代次数关系的图像
            test：是否自己选取测试用图像
        """
        return parameters
    if s==0:
        # 图片路径
        image_path = addr

        image = Image.open(image_path)

        # 将图像调整为 64x64 大小
        resized_image = image.resize((64, 64))

        # 将图像数据转换为 NumPy 数组
        image_array = np.array(resized_image)

        # 创建 HDF5 文件并保存图像数据、list_classes 和 train_set_y
        with h5py.File('output/output_image.h5', 'w') as h5_file:
            h5_file.create_dataset('output_x', data=image_array.reshape(1, 64, 64, 3))

        output_dataset = h5py.File('output/output_image.h5', "r")
        output_x_orig = np.array(output_dataset["output_x"][:])  # your train set features
        # 将图像的维度降低并转置。
        output_x_flatten = output_x_orig.reshape(output_x_orig.shape[0], -1).T
        output_x = output_x_flatten / 255
        probs = Predict(output_x,parameters)
        if probs[0][0] >= 0.5:
            print('是猫')
            return 1
        else:
            print("不是猫")
            return 0
