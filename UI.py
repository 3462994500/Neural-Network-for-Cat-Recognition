from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import *
from Deeplearningcat import switch
import sys

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        # MainWindow.setWindowIcon(QIcon('..\icons\zhenyanglog.png'))
        MainWindow.resize(1025, 600)
        MainWindow.setFixedSize(1025, 600)
        self.outputTextEdit = QtWidgets.QTextEdit(MainWindow)
        self.outputTextEdit.setGeometry(QtCore.QRect(620, 120, 300, 390))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 0, 1001, 31))
        self.label.setObjectName("label")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(10, 25, 1011, 21))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(10, 40, 91, 61))
        self.pushButton.setObjectName("pushButton")

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(100, 40, 91, 61))
        self.pushButton_2.setObjectName("pushButton_2")

        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(440, 54, 91, 31))
        self.pushButton_4.setObjectName("pushButton_4")

        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(320, 30, 110, 25))
        self.pushButton_5.setObjectName("pushButton_5")

        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(320, 55, 110, 25))
        self.pushButton_6.setObjectName("pushButton_6")

        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setGeometry(QtCore.QRect(320, 80, 110, 25))
        self.pushButton_7.setObjectName("pushButton_7")

        font3 = QFont()
        font3.setPointSize(17)


        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.pushButton_5.setEnabled(False)
        self.pushButton_6.setEnabled(False)
        self.pushButton_7.setEnabled(False)
        # self.pushButton_3.setEnabled(False)

        # 按钮关联函数
        self.pushButton.clicked.connect(self.ClickButton1)
        self.pushButton_2.clicked.connect(self.ClickButton2)
        self.pushButton_4.clicked.connect(self.openImage)
        self.pushButton_5.clicked.connect(self.ClickButton5)
        self.pushButton_6.clicked.connect(self.ClickButton6)
        self.pushButton_7.clicked.connect(self.ClickButton7)

        self.lineEdit_5 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_5.setGeometry(QtCore.QRect(200, 30, 130, 25))
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_6.setGeometry(QtCore.QRect(200, 55, 130, 25))
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.lineEdit_7 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_7.setGeometry(QtCore.QRect(200, 80, 130, 25))
        self.lineEdit_7.setObjectName("lineEdit_7")

        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(530, 54, 241, 31))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(10, 95, 1011, 21))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")

        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setGeometry(QtCore.QRect(-20, 109, 60, 486))
        self.line_5.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.line_6 = QtWidgets.QFrame(self.centralwidget)
        self.line_6.setGeometry(QtCore.QRect(-10, 34, 38, 71))
        self.line_6.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.line_7 = QtWidgets.QFrame(self.centralwidget)
        self.line_7.setGeometry(QtCore.QRect(1000, 34, 38, 71))
        self.line_7.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.line_8 = QtWidgets.QFrame(self.centralwidget)
        self.line_8.setGeometry(QtCore.QRect(970, 0, 98, 37))
        self.line_8.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.line_9 = QtWidgets.QFrame(self.centralwidget)
        self.line_9.setGeometry(QtCore.QRect(-10, 0, 38, 33))
        self.line_9.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_9.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_9.setObjectName("line_9")
        self.line_10 = QtWidgets.QFrame(self.centralwidget)
        self.line_10.setGeometry(QtCore.QRect(10, 10, 1011, 46))
        self.line_10.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_10.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_10.setObjectName("line_10")
        self.line_11 = QtWidgets.QFrame(self.centralwidget)
        self.line_11.setGeometry(QtCore.QRect(10, -9, 1011, 22))
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")

        self.line_13 = QtWidgets.QFrame(self.centralwidget)
        self.line_13.setGeometry(QtCore.QRect(10, 95, 801, 31))
        self.line_13.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_13.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_13.setObjectName("line_13")

        self.line_15 = QtWidgets.QFrame(self.centralwidget)
        self.line_15.setGeometry(QtCore.QRect(10, 566, 801, 61))
        self.line_15.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_15.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_15.setObjectName("line_15")
        self.line_16 = QtWidgets.QFrame(self.centralwidget)
        self.line_16.setGeometry(QtCore.QRect(920, 110, 41, 486))
        self.line_16.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_16.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_16.setObjectName("line_16")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 120, 300, 390))
        self.label_3.setStyleSheet("font:28px;\n"
                                   "border-style:solid;\n"
                                   "border-width:1px;\n"
                                   "border-color:rgb(45, 45, 45);\n"
                                   "\n"
                                   "")
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")

        self.label_30 = QtWidgets.QLabel(self.centralwidget)
        self.label_30.setGeometry(QtCore.QRect(320, 120, 300, 390))
        self.label_30.setStyleSheet("font:28px;\n"
                                   "border-style:solid;\n"
                                   "border-width:1px;\n"
                                   "border-color:rgb(45, 45, 45);\n"
                                   "\n"
                                   "")
        self.label_30.setText("")
        self.label_30.setObjectName("label_30")

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(780, 54, 191, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_5.setFont(font)
        self.label_5.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_5.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_5.setWordWrap(True)
        self.label_5.setObjectName("label_5")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(850, 54, 100, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(820, 280, 191, 41))

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def ClickButton1(self):
        self.lineEdit.clear()
        res = switch(0, imgNamepath,parameters,iterations, learning_rate)
        if res == 1:
            self.lineEdit.setText("是猫")
        else:
            self.lineEdit.setText("不是猫")
        # self.distance()

    def ClickButton2(self):
        global parameters
        parameters=switch(1, imgNamepath,0,iterations, learning_rate)
        img_path = 'datasets\plot.png'   # 图像路径
        pixmap = QPixmap(img_path)
        self.label_30.setPixmap(pixmap)
        self.label_30.setScaledContents(True)
        self.pushButton.setEnabled(True)

    def ClickButton5(self):
        global batch_size
        input_value1 = self.lineEdit_5.text()
        batch_size = int(input_value1)
        print(f'更新batch_size为: {batch_size}')
    def ClickButton6(self):
        global iterations
        input_value2 = self.lineEdit_6.text()
        iterations = int(input_value2)
        print(f'更新迭代次数为: {iterations}')

    def ClickButton7(self):
        global learning_rate
        input_value3 = self.lineEdit_7.text()
        learning_rate = float(input_value3)
        print(f'更新学习率为: {learning_rate}')

    # 系统目录方法
    def initUI(self, Qmodelidx):
        self.label_3.clear()
        self.label_30.clear()

    # 选择本地图片上传
    def openImage(self):
        global imgNamepath  # 这里为了方便别的地方引用图片路径，将其设置为全局变量

        imgNamepath, imgType = QFileDialog.getOpenFileName(self.centralwidget, "选择图片",
                                                           ".",
                                                           "All Files(*);;*.jpg;;*.png")
        # 通过文件路径获取图片文件，并设置图片长宽为label控件的长、宽
        # print(imgNamepath)
        img = QtGui.QPixmap(imgNamepath)  # .scaled(self.label_3.width(), self.label_3.height())
        # 在label控件上显示选择的图片

        self.label_3.setPixmap(img)
        self.label_3.setScaledContents(True)
        # 显示所选图片的路径
        self.lineEdit_4.setText(imgNamepath)

        self.pushButton_2.setEnabled(True)
        self.pushButton_5.setEnabled(True)
        self.pushButton_6.setEnabled(True)
        self.pushButton_7.setEnabled(True)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "图像识别"))
        self.label.setText(_translate("MainWindow", "识别UI界面"))
        self.pushButton.setText(_translate("MainWindow", "识别图像"))
        self.pushButton_2.setText(_translate("MainWindow", "训练与测试"))
        self.label_5.setText(_translate("MainWindow", "识别结果:"))
        self.pushButton_4.setText(_translate("MainWindow", "选择图片"))
        self.pushButton_5.setText(_translate("MainWindow", "batch_size"))
        self.pushButton_6.setText(_translate("MainWindow", "iterations"))
        self.pushButton_7.setText(_translate("MainWindow", "learning_rate"))
    def redirect_output(self):
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)

    def normalOutputWritten(self, text):
        cursor = self.outputTextEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.outputTextEdit.setTextCursor(cursor)  # 设置完文本后，重新设置光标位置
        self.outputTextEdit.ensureCursorVisible()

class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))


if __name__ == '__main__':
    # global iterations, learning_rate, batch_size
    iterations=1600
    learning_rate=0.02
    batch_size=10
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.redirect_output()
    MainWindow.show()
    sys.exit(app.exec_())