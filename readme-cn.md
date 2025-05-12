# Neural-Network-for-Cat-Recognition

For [English Readme](./readme.md).

本项目给出使用神经网络识别猫的神经网络代码，UI页面代码，综合报告

项目一共包含三个python文件：lr_utils.py，Deeplearningcat.py，UI.py。其中，lr_utils.py是由题目给出的，用于解析训练集和测试集文件的内容；Deeplearningcat.py是用于构建单层神经网络模型的文件；UI.py将实验结果可视化，并为用户提供交互接口。

首先，【吴恩达课后编程作业】第二周的作业数据集有209张图片作为训练集，50张图片作为测试集，图片中有的是猫的图片，有的不是。每张图片的像素大小为64*64，吴恩达把这两个图片集转换成训练集和测试集文件：train_catvnoncat.h5（训练集），test_catvnoncat.h5（测试集）。在作业给出的已知条件中有训练集文件、测试集文件和lr_utils.py文件。

实验效果如下：
![1651146092379.png](图片/UI.png "UI界面效果图")
