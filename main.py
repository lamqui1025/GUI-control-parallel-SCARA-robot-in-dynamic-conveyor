import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from qtGUI1 import Ui_MainWindow

from time import time
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.datasets import letterbox, LoadStreams, LoadImages
from utils.plots import plot_one_box

from threadSerial import serialThread

import torch
import torch.backends.cudnn as cudnn

import serial
import serial.tools.list_ports
import cv2
import numpy as np
import random

class ModelYolov7:
    source = 0
    device = ''
    weights = 'best-1.pt'
    image_size = 640
    conf_thres = 0.75
    iou_thres = 0.45
    view_img = False
    save_txt = False
    save_conf = False
    nosave = False
    classes = None
    agnostic_nms = False
    augment = False
    update = False
    no_trace = True
    trace = not no_trace

    def __init__(self):
        imgsz = self.image_size
        device = select_device(self.device)
        self.half = device.type != 'cpu'
        cudnn.benchmark = True
        self.model = attempt_load(weights=self.weights, map_location=device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if self.trace:
            self.model = TracedModel(self.model, device, self.image_size)

        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]


class MainWindow(QMainWindow):
    dictParity = {0: 'N', 1: 'E', 2: 'O', 3: 'M'}
    yolo_v7 = ModelYolov7()
    model = yolo_v7.model
    image_size = yolo_v7.image_size
    conf_thres = yolo_v7.conf_thres
    iou_thres = yolo_v7.iou_thres
    source = yolo_v7.source
    device = select_device(yolo_v7.device)
    half = yolo_v7.half
    augment = yolo_v7.augment
    classes = yolo_v7.classes
    agnostic_nms = yolo_v7.agnostic_nms

    names = yolo_v7.names
    colors = yolo_v7.colors

    def __init__(self):
        # self.main_win = QMainWindow()
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        # init for stream
        self.timer_video = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.out = None

        # khai bao nut an chay
        self.uic.btnConnect.clicked.connect(self.btnConnect_clicked)
        self.uic.btnStart.clicked.connect(self.start_capture_video)
        self.uic.btnStop.clicked.connect(self.stop_capture_video)
        self.timer_video.timeout.connect(self.show_video_frame)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.Timer_tick)
        self.timer.start(200)

        # self.serial1 = serial.Serial()
        # ports = serial.tools.list_ports.comports()
        # for port, desc, hwid in sorted(ports):
        #     self.uic.cbPort.addItem(port)

        self.serial1 = serialThread()
        ports = serial.tools.list_ports.comports()
        for port, desc, hwid in sorted(ports):
            self.uic.cbPort.addItem(port)
        # self.thread = {}

    def closeEvent(self, event):
        self.stop_capture_video()

    def stop_capture_video(self):
        self.timer_video.stop()
        self.uic.btnStart.setEnabled(True)
        self.cap.release()

    def start_capture_video(self):
        # self.thread[1] = capture_video(index=1)
        # self.thread[1].start()
        # self.thread[1].signal.connect(self.show_webcam)

        if not self.timer_video.isActive():
            flag = self.cap.open(self.source)
            if flag == False:
                QtWidgets.QMessageBox.warning(self, u"Warning", u"Webcam is not active!",
                                              buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                # self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(
                #     *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
                self.timer_video.start(30)
                self.uic.btnStart.setDisabled(True)

    # def show_webcam(self, frame):
    #     """Updates the image_label with a new opencv image"""
    #     qt_img = self.convert_cv_qt(frame)
    #     self.uic.lbScreen.setPixmap(qt_img)
    #
    # def convert_cv_qt(self, cv_img):
    #     """Convert from an opencv image to QPixmap"""
    #     rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    #     h, w, ch = rgb_image.shape
    #     bytes_per_line = ch * w
    #     convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    #     p = convert_to_Qt_format.scaled(800, 600, Qt.KeepAspectRatio)
    #     return QPixmap.fromImage(p)
    def show_video_frame(self):
        name_list = []
        flag, img = self.cap.read()

        if img is not None:
            showimg = img
            with torch.no_grad():
                img = letterbox(img, new_shape=self.image_size)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img, augment=self.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                           agnostic=self.agnostic_nms)
                print('pred : ', pred)
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            print(label)
                            plot_one_box(xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=2)
                            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            print('c1: ', c1, 'c2: ', c2)
                            center = (int((c1[0]+c2[0])/2), int((c1[1]+c2[1])/2))
                            cv2.circle(showimg, center, radius=0, color=(0,0,255), thickness=3)

            # self.out.write(showimg)
            # show = showimg
            show = cv2.resize(showimg, (640, 480))

            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            # showImage = showImage.scaled(700, 550, Qt.KeepAspectRatio)
            self.uic.lbScreen.setPixmap(QtGui.QPixmap.fromImage(showImage))

        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.uic.lbScreen.clear()
            self.uic.btnStart.setDisabled(False)
    # ----------------------------------

    def btnConnect_clicked(self):
        if self.serial1.serialPort.is_open():
            self.ClosePort()
        else:
            self.OpenPort()

    def OpenPort(self):
        portname = self.uic.cbPort.currentText()
        baud = int(self.uic.cbBaud.currentText())
        parity = self.dictParity[self.uic.cbParity.currentIndex()]

        print("name:", self.serial1.serialPort.name, "---baud:", self.serial1.serialPort.baudrate, "---Parity:", self.serial1.serialPort.parity)
        self.serial1.serialPort.setPort(portname)
        self.serial1.serialPort.baudrate = baud
        self.serial1.serialPort.parity = parity
        print("name:", self.serial1.serialPort.name, "---baud:", self.serial1.serialPort.baudrate, "---Parity:", serial.PARITY_NAMES[parity])

        self.serial1.serialPort.serialPort.open()
        self.serial1.serialPort.write('ABCD'.encode('utf-8'))
        # self.serial1.close()

    def ClosePort(self):
        self.serial1.serialPort.close()

    def Timer_tick(self):
        if self.serial1.serialPort.is_open:
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

class ThreadSerial(QThread):
    signal = pyqtSignal(int)

    def __init__(self, index=0):
        super().__init__()
        self.index = index

    def run(self):
        print('Starting Thread...', self.index)


# class capture_video(QThread):
#     signal = pyqtSignal(np.ndarray)

    # def __init__(self, index):
    #     self.index = index
    #     print("start threading", self.index)
    #     super(capture_video, self).__init__()

    # def run(self):
    #
    #     self.model = self.load_model()
    #     self.classes = self.model.names
    #     self.out_file = "Labeled_Video.avi"
    #     self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     self.run_program()

        # cap = cv2.VideoCapture(0)  # 'D:/8.Record video/My Video.mp4'
        # while True:
        #     ret, cv_img = cap.read()
        #     if ret:
        #         self.signal.emit(cv_img)

    # def get_video_from_url(self):
    #     return cv2.VideoCapture(0)
    #
    # def load_model(self):
    #     model = torch.hub.load('WongKinYiu/yolov7', 'yolov7', pretrained=True)
    #     return model
    #
    # def score_frame(self, frame):
    #     self.model.to(self.device)
    #     frame = [frame]
    #     results = self.model(frame)
    #     labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    #     return labels, cord
    #
    # def class_to_label(self, x):
    #     return self.classes[int(x)]

    # def plot_boxes(self, results, frame):
    #     labels, cord = results
    #     n = len(labels)
    #     x_shape, y_shape = frame.shape[1], frame.shape[0]
    #     for i in range(n):
    #         row = cord[i]
    #         if row[4] >= confident:
    #             x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
    #             bgr = (0, 255, 0)
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
    #             cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    #     return frame

    # def run_program(self):
    #     player = self.get_video_from_url()
    #     assert player.isOpened()
    #     x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     # four_cc = cv2.VideoWriter_fourcc(*"MJPG")
    #     # out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))
    #     while True:
    #         start_time = time()
    #         ret, frame = player.read()
    #         assert ret
    #         results = self.score_frame(frame)
    #         frame = self.plot_boxes(results, frame)
    #         end_time = time()
    #         fps = 1 / (np.round(end_time - start_time, 3))
    #         print((f"Frame Per Second : {round(fps, 3, )} FPS"))
    #         self.signal.emit(frame)

    # def stop(self):
    #     print("stop threading", self.index)
    #     self.terminate()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
