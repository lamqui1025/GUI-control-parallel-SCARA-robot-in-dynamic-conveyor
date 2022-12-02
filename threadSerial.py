import serial
from PyQt5.QtCore import pyqtSignal, QThread, QObject
from PyQt5 import QtSerialPort, QtCore
import time

class serialThread(QThread):
    message = pyqtSignal(bytes)

    def __init__(self, parent=None):
        super(serialThread, self).__init__(parent)
        self.serialPort = QtSerialPort.QSerialPort()
        self.serialPort.readyRead.connect(self.receive)
        # self.serialPort.baudrate = baudrate
        # self.serialPort.port = port
        # self.serialPort.open()

    # def run(self):
    #     while self.serialPort.canReadLine():
    #         veri = self.serialPort.readLine().data().decode()
    #         print(veri)
    #         self.message.emit(str(veri))

    def receive(self):
        # while True:
        #     rec = self.serialPort.readAll()
        #     print(rec)
        #     self.message.emit(str(rec))
        rec = self.serialPort.readAll()
        self.message.emit(bytes(rec))
    def Conf(self, portName, baudrate, parity):
        self.serialPort.setPortName(portName)
        self.serialPort.setBaudRate(baudrate)
        self.serialPort.setParity(parity)

    def Open(self):
        if not self.serialPort.isOpen():
            self.serialPort.open(QtCore.QIODevice.ReadWrite)

    def Close(self):
        if self.serialPort.isOpen():
            self.serialPort.close()

    def is_Open(self):
        return self.serialPort.isOpen()

    def sendSerial(self, buff):
        n = self.serialPort.write(buff)
        return n
    # def work(self):
    #     while self.working:
    #         rec = self.serialPort.read().decode('utf-8')
    #         print(rec)
    #         self.message.emit(str(rec))
    #
    #
    # def stop(self):
    #     self.working = False
