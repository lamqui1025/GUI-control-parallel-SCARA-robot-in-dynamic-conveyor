#!/usr/bin/python3
import sys
from PyQt5 import QtCore, QtWidgets, QtSerialPort


def ports():
    availables = []
    for port in QtSerialPort.QSerialPortInfo.availablePorts():
        availables.append(port.systemLocation())
    if availables:
        availables = str(availables)
        return (availables.translate({ord(c): None for c in "[']"}))
    return ("none")


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    print(ports())
    sys.exit(app.exec_())