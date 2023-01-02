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
        MainWindow.resize(1110, 852)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lbScreen = QtWidgets.QLabel(self.centralwidget)
        self.lbScreen.setGeometry(QtCore.QRect(230, 10, 641, 481))
        self.lbScreen.setFrameShape(QtWidgets.QFrame.Box)
        self.lbScreen.setLineWidth(2)
        self.lbScreen.setText("")
        self.lbScreen.setObjectName("lbScreen")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 191, 241))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox.setFont(font)
        self.groupBox.setAutoFillBackground(True)
        self.groupBox.setObjectName("groupBox")
        self.btnConnect = QtWidgets.QPushButton(self.groupBox)
        self.btnConnect.setGeometry(QtCore.QRect(40, 180, 101, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btnConnect.setFont(font)
        self.btnConnect.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.btnConnect.setObjectName("btnConnect")
        self.lbStatus = QtWidgets.QLabel(self.groupBox)
        self.lbStatus.setGeometry(QtCore.QRect(40, 150, 111, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lbStatus.setFont(font)
        self.lbStatus.setObjectName("lbStatus")
        self.layoutWidget = QtWidgets.QWidget(self.groupBox)
        self.layoutWidget.setGeometry(QtCore.QRect(13, 28, 165, 106))
        self.layoutWidget.setObjectName("layoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.layoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.lbPort = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lbPort.setFont(font)
        self.lbPort.setObjectName("lbPort")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.lbPort)
        self.cbPort = QtWidgets.QComboBox(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.cbPort.setFont(font)
        self.cbPort.setObjectName("cbPort")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.cbPort)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.cbBaud = QtWidgets.QComboBox(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.cbBaud.setFont(font)
        self.cbBaud.setObjectName("cbBaud")
        self.cbBaud.addItem("")
        self.cbBaud.addItem("")
        self.cbBaud.addItem("")
        self.cbBaud.addItem("")
        self.cbBaud.addItem("")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.cbBaud)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.cbParity = QtWidgets.QComboBox(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.cbParity.setFont(font)
        self.cbParity.setObjectName("cbParity")
        self.cbParity.addItem("")
        self.cbParity.addItem("")
        self.cbParity.addItem("")
        self.cbParity.addItem("")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.cbParity)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 610, 181, 191))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.lbX = QtWidgets.QLabel(self.groupBox_2)
        self.lbX.setGeometry(QtCore.QRect(14, 52, 16, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lbX.setFont(font)
        self.lbX.setObjectName("lbX")
        self.lbY = QtWidgets.QLabel(self.groupBox_2)
        self.lbY.setGeometry(QtCore.QRect(14, 79, 16, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lbY.setFont(font)
        self.lbY.setObjectName("lbY")
        self.lbZ = QtWidgets.QLabel(self.groupBox_2)
        self.lbZ.setGeometry(QtCore.QRect(14, 106, 16, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lbZ.setFont(font)
        self.lbZ.setObjectName("lbZ")
        self.lbtheta1 = QtWidgets.QLabel(self.groupBox_2)
        self.lbtheta1.setGeometry(QtCore.QRect(14, 133, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lbtheta1.setFont(font)
        self.lbtheta1.setObjectName("lbtheta1")
        self.lbtheta4 = QtWidgets.QLabel(self.groupBox_2)
        self.lbtheta4.setGeometry(QtCore.QRect(14, 160, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lbtheta4.setFont(font)
        self.lbtheta4.setObjectName("lbtheta4")
        self.lb_vX = QtWidgets.QLabel(self.groupBox_2)
        self.lb_vX.setGeometry(QtCore.QRect(94, 52, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.lb_vX.setFont(font)
        self.lb_vX.setObjectName("lb_vX")
        self.lb_vY = QtWidgets.QLabel(self.groupBox_2)
        self.lb_vY.setGeometry(QtCore.QRect(94, 79, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.lb_vY.setFont(font)
        self.lb_vY.setObjectName("lb_vY")
        self.lb_vZ = QtWidgets.QLabel(self.groupBox_2)
        self.lb_vZ.setGeometry(QtCore.QRect(94, 106, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.lb_vZ.setFont(font)
        self.lb_vZ.setObjectName("lb_vZ")
        self.lb_vtheta1 = QtWidgets.QLabel(self.groupBox_2)
        self.lb_vtheta1.setGeometry(QtCore.QRect(94, 133, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.lb_vtheta1.setFont(font)
        self.lb_vtheta1.setObjectName("lb_vtheta1")
        self.lb_vtheta4 = QtWidgets.QLabel(self.groupBox_2)
        self.lb_vtheta4.setGeometry(QtCore.QRect(94, 160, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.lb_vtheta4.setFont(font)
        self.lb_vtheta4.setObjectName("lb_vtheta4")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(230, 650, 581, 151))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.btnStop_robot = QtWidgets.QPushButton(self.groupBox_3)
        self.btnStop_robot.setGeometry(QtCore.QRect(140, 80, 101, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.btnStop_robot.setFont(font)
        self.btnStop_robot.setObjectName("btnStop_robot")
        self.btnStart_robot = QtWidgets.QPushButton(self.groupBox_3)
        self.btnStart_robot.setGeometry(QtCore.QRect(40, 80, 101, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.btnStart_robot.setFont(font)
        self.btnStart_robot.setObjectName("btnStart_robot")
        self.btnReset_robot = QtWidgets.QPushButton(self.groupBox_3)
        self.btnReset_robot.setGeometry(QtCore.QRect(40, 30, 201, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.btnReset_robot.setFont(font)
        self.btnReset_robot.setObjectName("btnReset_robot")
        self.lb_vmax = QtWidgets.QLabel(self.groupBox_3)
        self.lb_vmax.setGeometry(QtCore.QRect(460, 50, 61, 21))
        self.lb_vmax.setObjectName("lb_vmax")
        self.lb_amax = QtWidgets.QLabel(self.groupBox_3)
        self.lb_amax.setGeometry(QtCore.QRect(460, 95, 61, 21))
        self.lb_amax.setObjectName("lb_amax")
        self.spinBox_vmax = QtWidgets.QSpinBox(self.groupBox_3)
        self.spinBox_vmax.setGeometry(QtCore.QRect(360, 50, 81, 31))
        self.spinBox_vmax.setObjectName("spinBox_vmax")
        self.spinBox_amax = QtWidgets.QSpinBox(self.groupBox_3)
        self.spinBox_amax.setGeometry(QtCore.QRect(360, 90, 81, 31))
        self.spinBox_amax.setObjectName("spinBox_amax")
        self.label = QtWidgets.QLabel(self.groupBox_3)
        self.label.setGeometry(QtCore.QRect(300, 50, 51, 31))
        self.label.setObjectName("label")
        self.label_4 = QtWidgets.QLabel(self.groupBox_3)
        self.label_4.setGeometry(QtCore.QRect(300, 90, 51, 31))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.groupBox_3)
        self.label_5.setGeometry(QtCore.QRect(520, 50, 55, 21))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.groupBox_3)
        self.label_6.setGeometry(QtCore.QRect(520, 95, 55, 21))
        self.label_6.setObjectName("label_6")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(10, 260, 191, 251))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setAutoFillBackground(True)
        self.groupBox_4.setObjectName("groupBox_4")
        self.lbPort_2 = QtWidgets.QLabel(self.groupBox_4)
        self.lbPort_2.setGeometry(QtCore.QRect(20, 20, 51, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lbPort_2.setFont(font)
        self.lbPort_2.setObjectName("lbPort_2")
        self.cbPort_conveyor = QtWidgets.QComboBox(self.groupBox_4)
        self.cbPort_conveyor.setGeometry(QtCore.QRect(81, 20, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.cbPort_conveyor.setFont(font)
        self.cbPort_conveyor.setObjectName("cbPort_conveyor")
        self.btnConnect_conveyor = QtWidgets.QPushButton(self.groupBox_4)
        self.btnConnect_conveyor.setGeometry(QtCore.QRect(40, 60, 101, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btnConnect_conveyor.setFont(font)
        self.btnConnect_conveyor.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.btnConnect_conveyor.setObjectName("btnConnect_conveyor")
        self.btnRun_conveyor = QtWidgets.QPushButton(self.groupBox_4)
        self.btnRun_conveyor.setGeometry(QtCore.QRect(10, 200, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btnRun_conveyor.setFont(font)
        self.btnRun_conveyor.setObjectName("btnRun_conveyor")
        self.spinBox_Vel = QtWidgets.QSpinBox(self.groupBox_4)
        self.spinBox_Vel.setGeometry(QtCore.QRect(11, 120, 81, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.spinBox_Vel.setFont(font)
        self.spinBox_Vel.setMaximum(999)
        self.spinBox_Vel.setObjectName("spinBox_Vel")
        self.btnStop_conveyor = QtWidgets.QPushButton(self.groupBox_4)
        self.btnStop_conveyor.setGeometry(QtCore.QRect(100, 200, 81, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btnStop_conveyor.setFont(font)
        self.btnStop_conveyor.setObjectName("btnStop_conveyor")
        self.lb_realVel = QtWidgets.QLabel(self.groupBox_4)
        self.lb_realVel.setGeometry(QtCore.QRect(21, 170, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lb_realVel.setFont(font)
        self.lb_realVel.setObjectName("lb_realVel")
        self.label_7 = QtWidgets.QLabel(self.groupBox_4)
        self.label_7.setGeometry(QtCore.QRect(124, 130, 51, 51))
        self.label_7.setObjectName("label_7")
        self.btnScaleCam = QtWidgets.QPushButton(self.centralwidget)
        self.btnScaleCam.setGeometry(QtCore.QRect(890, 450, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btnScaleCam.setFont(font)
        self.btnScaleCam.setObjectName("btnScaleCam")
        self.btnYolo_work = QtWidgets.QPushButton(self.centralwidget)
        self.btnYolo_work.setGeometry(QtCore.QRect(990, 450, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btnYolo_work.setFont(font)
        self.btnYolo_work.setObjectName("btnYolo_work")
        self.tb_clsObjs = QtWidgets.QTableWidget(self.centralwidget)
        self.tb_clsObjs.setGeometry(QtCore.QRect(230, 500, 871, 151))
        self.tb_clsObjs.setObjectName("tb_clsObjs")
        self.tb_clsObjs.setColumnCount(0)
        self.tb_clsObjs.setRowCount(0)
        self.lsw_stack_obj = QtWidgets.QListWidget(self.centralwidget)
        self.lsw_stack_obj.setGeometry(QtCore.QRect(890, 10, 211, 401))
        self.lsw_stack_obj.setObjectName("lsw_stack_obj")
        self.lb_numof_missed = QtWidgets.QLabel(self.centralwidget)
        self.lb_numof_missed.setGeometry(QtCore.QRect(990, 422, 61, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.lb_numof_missed.setFont(font)
        self.lb_numof_missed.setObjectName("lb_numof_missed")
        self.lb_textMissed = QtWidgets.QLabel(self.centralwidget)
        self.lb_textMissed.setGeometry(QtCore.QRect(900, 420, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lb_textMissed.setFont(font)
        self.lb_textMissed.setObjectName("lb_textMissed")
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
        self.groupBox.setTitle(_translate("MainWindow", "Serial"))
        self.btnConnect.setText(_translate("MainWindow", "Open"))
        self.lbStatus.setText(_translate("MainWindow", "Disconnected"))
        self.lbPort.setText(_translate("MainWindow", "Port"))
        self.label_2.setText(_translate("MainWindow", "Baud"))
        self.cbBaud.setItemText(0, _translate("MainWindow", "9600"))
        self.cbBaud.setItemText(1, _translate("MainWindow", "14400"))
        self.cbBaud.setItemText(2, _translate("MainWindow", "19200"))
        self.cbBaud.setItemText(3, _translate("MainWindow", "56000"))
        self.cbBaud.setItemText(4, _translate("MainWindow", "115200"))
        self.label_3.setText(_translate("MainWindow", "Parity"))
        self.cbParity.setItemText(0, _translate("MainWindow", "none"))
        self.cbParity.setItemText(1, _translate("MainWindow", "even"))
        self.cbParity.setItemText(2, _translate("MainWindow", "odd"))
        self.cbParity.setItemText(3, _translate("MainWindow", "mark"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Current Position"))
        self.lbX.setText(_translate("MainWindow", "X"))
        self.lbY.setText(_translate("MainWindow", "Y"))
        self.lbZ.setText(_translate("MainWindow", "Z"))
        self.lbtheta1.setText(_translate("MainWindow", "Theta1"))
        self.lbtheta4.setText(_translate("MainWindow", "Theta4"))
        self.lb_vX.setText(_translate("MainWindow", "0"))
        self.lb_vY.setText(_translate("MainWindow", "0"))
        self.lb_vZ.setText(_translate("MainWindow", "0"))
        self.lb_vtheta1.setText(_translate("MainWindow", "0"))
        self.lb_vtheta4.setText(_translate("MainWindow", "0"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Control"))
        self.btnStop_robot.setText(_translate("MainWindow", "Stop"))
        self.btnStart_robot.setText(_translate("MainWindow", "Start"))
        self.btnReset_robot.setText(_translate("MainWindow", "Reset robot"))
        self.lb_vmax.setText(_translate("MainWindow", "0"))
        self.lb_amax.setText(_translate("MainWindow", "0"))
        self.label.setText(_translate("MainWindow", "vMax"))
        self.label_4.setText(_translate("MainWindow", "aMax"))
        self.label_5.setText(_translate("MainWindow", "deg/s"))
        self.label_6.setText(_translate("MainWindow", "deg/s2"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Conveyor"))
        self.lbPort_2.setText(_translate("MainWindow", "Port"))
        self.btnConnect_conveyor.setText(_translate("MainWindow", "Open"))
        self.btnRun_conveyor.setText(_translate("MainWindow", "RUN"))
        self.btnStop_conveyor.setText(_translate("MainWindow", "STOP"))
        self.lb_realVel.setText(_translate("MainWindow", "0"))
        self.label_7.setText(_translate("MainWindow", "mm/s"))
        self.btnScaleCam.setText(_translate("MainWindow", "Scale"))
        self.btnYolo_work.setText(_translate("MainWindow", "START"))
        self.lb_numof_missed.setText(_translate("MainWindow", "0"))
        self.lb_textMissed.setText(_translate("MainWindow", "Missed"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
