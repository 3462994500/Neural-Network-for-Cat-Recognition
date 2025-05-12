# import h5py
# import numpy as np
# from PIL import Image
#
# # # 图片路径
# # image_path = "E:\life\Camera_XHS_17099921877121040g00830sspci3kjs005nr9rjt09e1sjf8shv0.jpg"
# #
# # image = Image.open(image_path)
# #
# # # 将图像调整为 64x64 大小
# # resized_image = image.resize((64, 64))
# #
# # # 将图像数据转换为 NumPy 数组
# # image_array = np.array(resized_image)
# #
# # # 创建 HDF5 文件并保存图像数据、list_classes 和 train_set_y
# # with h5py.File('datasets/output_image.h5', 'w') as h5_file:
# #     h5_file.create_dataset('output_x', data=image_array.reshape(1, 64, 64, 3))
# #
# #
# # # 打开HDF5文件
# # file = h5py.File('datasets/output_image.h5', 'r')
# #
# # # 读取数据集内容
# # output_x = file['output_x'][:]
# #
# #
# # # 打印数据集内容
# # print("output_x shape:", output_x.shape)
# #
# # # 关闭HDF5文件
# # file.close()
#
# train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
#
# print(train_dataset["train_set_x"][:].shape)
# train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
# train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels
# print(train_set_x_orig.shape)
# print(train_set_y_orig)
# print(train_set_y_orig.shape)
# train_dataset.close()
#
# train_dataset = h5py.File('datasets/output_image.h5', "r")
#
# print(train_dataset["output_x"][:].shape)
# # train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
# # print(train_set_x_orig.shape)
#
import sys
from PyQt5 import QtWidgets, QtGui, QtCore

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(400, 300)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.outputTextEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.outputTextEdit.setObjectName("outputTextEdit")
        self.verticalLayout.addWidget(self.outputTextEdit)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Click Me"))

class MyMainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.redirect_output()

        # 连接按钮的点击事件到相应的槽函数
        self.pushButton.clicked.connect(self.on_button_clicked)

    def on_button_clicked(self):
        # 当按钮被点击时，输出信息到 QTextEdit 控件中
        print("Button clicked!")

    def redirect_output(self):
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)

    def normalOutputWritten(self, text):
        cursor = self.outputTextEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.outputTextEdit.setTextCursor(cursor)
        self.outputTextEdit.ensureCursorVisible()

class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MyMainWindow()
    MainWindow.show()
    sys.exit(app.exec_())
