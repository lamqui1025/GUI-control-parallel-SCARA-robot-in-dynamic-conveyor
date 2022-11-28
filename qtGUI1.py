# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qt5gui1.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1110, 864)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lbScreen = QtWidgets.QLabel(self.centralwidget)
        self.lbScreen.setGeometry(QtCore.QRect(220, 20, 640, 480))
        self.lbScreen.setFrameShape(QtWidgets.QFrame.Box)
        self.lbScreen.setLineWidth(2)
        self.lbScreen.setText("")
        self.lbScreen.setObjectName("lbScreen")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 161, 241))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox.setFont(font)
        self.groupBox.setAutoFillBackground(True)
        self.groupBox.setObjectName("groupBox")
        self.btnConnect = QtWidgets.QPushButton(self.groupBox)
        self.btnConnect.setGeometry(QtCore.QRect(30, 186, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btnConnect.setFont(font)
        self.btnConnect.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.btnConnect.setObjectName("btnConnect")
        self.lbStatus = QtWidgets.QLabel(self.groupBox)
        self.lbStatus.setGeometry(QtCore.QRect(30, 150, 111, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lbStatus.setFont(font)
        self.lbStatus.setObjectName("lbStatus")
        self.widget = QtWidgets.QWidget(self.groupBox)
        self.widget.setGeometry(QtCore.QRect(59, 26, 91, 111))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.cbPort = QtWidgets.QComboBox(self.widget)
        self.cbPort.setObjectName("cbPort")
        self.verticalLayout.addWidget(self.cbPort)
        self.cbBaud = QtWidgets.QComboBox(self.widget)
        self.cbBaud.setObjectName("cbBaud")
        self.cbBaud.addItem("")
        self.cbBaud.addItem("")
        self.cbBaud.addItem("")
        self.cbBaud.addItem("")
        self.cbBaud.addItem("")
        self.verticalLayout.addWidget(self.cbBaud)
        self.cbParity = QtWidgets.QComboBox(self.widget)
        self.cbParity.setObjectName("cbParity")
        self.cbParity.addItem("")
        self.cbParity.addItem("")
        self.cbParity.addItem("")
        self.cbParity.addItem("")
        self.verticalLayout.addWidget(self.cbParity)
        self.widget1 = QtWidgets.QWidget(self.groupBox)
        self.widget1.setGeometry(QtCore.QRect(11, 26, 51, 111))
        self.widget1.setObjectName("widget1")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget1)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.lbPort = QtWidgets.QLabel(self.widget1)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lbPort.setFont(font)
        self.lbPort.setObjectName("lbPort")
        self.verticalLayout_2.addWidget(self.lbPort)
        self.label_2 = QtWidgets.QLabel(self.widget1)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.label_3 = QtWidgets.QLabel(self.widget1)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.btnStart = QtWidgets.QPushButton(self.centralwidget)
        self.btnStart.setGeometry(QtCore.QRect(490, 600, 111, 61))
        self.btnStart.setObjectName("btnStart")
        self.btnStop = QtWidgets.QPushButton(self.centralwidget)
        self.btnStop.setGeometry(QtCore.QRect(680, 600, 111, 61))
        self.btnStop.setObjectName("btnStop")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1110, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Serial"))
        self.btnConnect.setText(_translate("MainWindow", "Open"))
        self.lbStatus.setText(_translate("MainWindow", "Disconnected"))
        self.cbBaud.setItemText(0, _translate("MainWindow", "9600"))
        self.cbBaud.setItemText(1, _translate("MainWindow", "14400"))
        self.cbBaud.setItemText(2, _translate("MainWindow", "19200"))
        self.cbBaud.setItemText(3, _translate("MainWindow", "56000"))
        self.cbBaud.setItemText(4, _translate("MainWindow", "115200"))
        self.cbParity.setItemText(0, _translate("MainWindow", "none"))
        self.cbParity.setItemText(1, _translate("MainWindow", "even"))
        self.cbParity.setItemText(2, _translate("MainWindow", "odd"))
        self.cbParity.setItemText(3, _translate("MainWindow", "mark"))
        self.lbPort.setText(_translate("MainWindow", "Port"))
        self.label_2.setText(_translate("MainWindow", "Baud"))
        self.label_3.setText(_translate("MainWindow", "Parity"))
        self.btnStart.setText(_translate("MainWindow", "START"))
        self.btnStop.setText(_translate("MainWindow", "STOP"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
