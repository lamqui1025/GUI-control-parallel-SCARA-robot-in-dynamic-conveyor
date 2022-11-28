import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from qtGUI1 import Ui_MainWindow

from time import time

import torch
import serial
import serial.tools.list_ports
import cv2
import numpy as np

confident = 0.2

class MainWindow(QMainWindow):
    dictParity = {0: 'N', 1: 'E', 2: 'O', 3: 'M'}

    def __init__(self):
        # self.main_win = QMainWindow()
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        # khai bao nut an chay
        self.uic.btnConnect.clicked.connect(self.btnConnect_clicked)
        self.uic.btnStart.clicked.connect(self.start_capture_video)
        self.uic.btnStop.clicked.connect(self.stop_capture_video)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.Timer_tick)
        self.timer.start(200)

        self.serial1 = serial.Serial()
        ports = serial.tools.list_ports.comports()
        for port, desc, hwid in sorted(ports):
            self.uic.cbPort.addItem(port)

        self.thread = {}

    def closeEvent(self, event):
        self.stop_capture_video()

    def stop_capture_video(self):
        self.thread[1].stop()

    def start_capture_video(self):
        self.thread[1] = capture_video(index=1)
        self.thread[1].start()
        self.thread[1].signal.connect(self.show_webcam)

    def show_webcam(self, frame):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(frame)
        self.uic.lbScreen.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(800, 600, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    # ----------------------------------
    # def show(self):
    #     self.main_win.show()

    def btnConnect_clicked(self):
        if self.serial1.is_open:
            self.ClosePort()
        else:
            self.OpenPort()

    def OpenPort(self):
        portname = self.uic.cbPort.currentText()
        baud = int(self.uic.cbBaud.currentText())
        parity = self.dictParity[self.uic.cbParity.currentIndex()]

        print("name:", self.serial1.name, "---baud:", self.serial1.baudrate, "---Parity:", self.serial1.parity)
        self.serial1.setPort(portname)
        self.serial1.baudrate = baud
        self.serial1.parity = parity
        print("name:", self.serial1.name, "---baud:", self.serial1.baudrate, "---Parity:", serial.PARITY_NAMES[parity])

        self.serial1.open()
        self.serial1.write('ABCD'.encode('utf-8'))
        # self.serial1.close()

    def ClosePort(self):
        self.serial1.close()

    def Timer_tick(self):
        if self.serial1.is_open:
            self.uic.btnConnect.setText('Close')
            self.uic.lbStatus.setText('Connected')
            self.uic.lbStatus.setStyleSheet("QLabel {color : green; }")
            self.uic.cbPort.setDisabled(True)
            self.uic.cbBaud.setDisabled(True)
            self.uic.cbParity.setDisabled(True)
        else:
            self.uic.btnConnect.setText('Open')
            self.uic.lbStatus.setText('Disconnected')
            self.uic.lbStatus.setStyleSheet("QLabel {color : red; }")
            self.uic.cbPort.setEnabled(True)
            self.uic.cbBaud.setEnabled(True)
            self.uic.cbParity.setEnabled(True)


class capture_video(QThread):
    signal = pyqtSignal(np.ndarray)

    def __init__(self, index):
        self.index = index
        print("start threading", self.index)
        super(capture_video, self).__init__()

    def run(self):

        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = "Labeled_Video.avi"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.run_program()

        # cap = cv2.VideoCapture(0)  # 'D:/8.Record video/My Video.mp4'
        # while True:
        #     ret, cv_img = cap.read()
        #     if ret:
        #         self.signal.emit(cv_img)

    def get_video_from_url(self):
        return cv2.VideoCapture(0)

    def load_model(self):
        model = torch.hub.load('WongKinYiu/yolov7', 'yolov7', path='yolov7/yolov7.pt')
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= confident:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame

    def run_program(self):
        player = self.get_video_from_url()
        assert player.isOpened()
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        # out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))
        while True:
            start_time = time()
            ret, frame = player.read()
            assert ret
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps = 1 / (np.round(end_time - start_time, 3))
            print((f"Frame Per Second : {round(fps, 3, )} FPS"))
            self.signal.emit(frame)

    def stop(self):
        print("stop threading", self.index)
        self.terminate()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
