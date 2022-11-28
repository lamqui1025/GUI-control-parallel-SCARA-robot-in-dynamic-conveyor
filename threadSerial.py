import serial
from PyQt5.QtCore import pyqtSignal, QThread

class serialThread(QThread):
    message = pyqtSignal(str)

    def __init__(self, parent=None, port='', baudrate = 9600):
        super(serialThread, self).__init__(parent)
        self.serialPort = serial.Serial()
        # self.serialPort.baudrate = baudrate
        # self.serialPort.port = port
        # self.serialPort.open()

    def run(self):
        while True:
            veri = self.serialPort.readline()
            self.message.emit(str(veri))
            print(veri)

    def Setting(self, portName, baudrate, parity):
        self.serialPort.port = portName
        self.serialPort.baudrate = baudrate
        self.serialPort.parity = parity

    def Open(self):
        if not self.serialPort.is_open():
            self.serialPort.open()

    def Close(self):
        if self.serialPort.is_open():
            self.serialPort.close()

    def is_Open(self):
        return self.serialPort.is_open

    def sendSerial(self, buff):
        self.serialPort.write(buff)
