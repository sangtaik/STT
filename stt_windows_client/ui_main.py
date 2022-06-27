# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_main.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(993, 692)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pbLevel = QtWidgets.QProgressBar(self.centralwidget)
        self.pbLevel.setMaximum(1000)
        self.pbLevel.setProperty("value", 123)
        self.pbLevel.setTextVisible(False)
        self.pbLevel.setOrientation(QtCore.Qt.Vertical)
        self.pbLevel.setObjectName("pbLevel")
        self.horizontalLayout.addWidget(self.pbLevel)
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.textBrowser = QtWidgets.QTextBrowser(self.frame)  
        self.textBrowser.setGeometry(QtCore.QRect(0, 0, 692, 300))
        self.textBrowser.setObjectName("textBrowser")
#         self.textBrowser.setFixedWidth(100)
        self.verticalLayout.addWidget(self.textBrowser)
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.grPCM = PlotWidget(self.frame)
        self.grPCM.setObjectName("grPCM")
        self.grPCM.setHidden(False)
        self.verticalLayout.addWidget(self.grPCM)
        self.textMessage = QtWidgets.QTextBrowser(self.frame)  
        self.textMessage.setGeometry(QtCore.QRect(0, 0, 692, 200))
        self.textMessage.setObjectName("textMessage")
        self.textMessage.setHidden(True)
        self.verticalLayout.addWidget(self.textMessage)
        self.start_button = QtWidgets.QPushButton("음성인식 시작(Start)")
        self.stop_button = QtWidgets.QPushButton("음성인식 종료(Stop)")
        self.test_button = QtWidgets.QPushButton("명령어 동작 테스트(Command List Test)")
        self.verticalLayout.addWidget(self.start_button)
        self.verticalLayout.addWidget(self.stop_button)
        self.verticalLayout.addWidget(self.test_button)
        self.horizontalLayout.addWidget(self.frame)

        #  time Label
        self.label_time = QtWidgets.QLabel(self.frame)
        self.label_time.setObjectName("label_time")
        self.verticalLayout.addWidget(self.label_time)
        
        
        MainWindow.setCentralWidget(self.centralwidget)

        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "PPT 도우미"))
        self.textBrowser.setText(_translate("MainWindow", "audio to text"))
        self.label_2.setText(_translate("MainWindow", "raw data (PCM):"))
        self.label_time.setText(_translate("MainWindow", "Time to text"))
        
    
from pyqtgraph import PlotWidget
