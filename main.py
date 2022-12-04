import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets, QtSerialPort
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from qtGUI1 import Ui_MainWindow

import time
# from time import time
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.datasets import letterbox, LoadStreams, LoadImages
from utils.plots import plot_one_box

from sort import *
from threadSerial import serialThread

import torch
import torch.backends.cudnn as cudnn

import serial
import serial.tools.list_ports
import cv2
import numpy as np
import random

class ModelYolov7:
    source = 1
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
    track = True

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

        self.sort_tracker = Sort(max_age=5,
                            min_hits=2,
                            iou_threshold=0.2)

    def time_synchronized(self):
        # pytorch-accurate time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

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
    track = yolo_v7.track

    sort_tracker = yolo_v7.sort_tracker
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
        self.uic.btnConnect_conveyor.clicked.connect(self.btnConnect_conveyor_clicked)
        self.uic.btnStart.clicked.connect(self.start_capture_video)
        self.uic.btnStop.clicked.connect(self.stop_capture_video)
        self.uic.btnRun_conveyor.clicked.connect(self.btnRun_conveyor_clicked)
        self.uic.btnStop_conveyor.clicked.connect(self.btnStop_conveyor_clicked)
        self.uic.btnReset_robot.clicked.connect(self.btnReset_robot_clicked)
        self.uic.btnStart_robot.clicked.connect(self.btnStart_robot_clicked)
        self.uic.btnStop_robot.clicked.connect(self.btnStop_robot_clicked)

        self.timer_video.timeout.connect(self.show_video_frame)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.Timer_tick)
        self.timer.start(200)

        self.ser = {}
        self.ser[0] = serialThread()
        self.ser[1] = serialThread()
        ports = serial.tools.list_ports.comports()
        for port, desc, hwid in sorted(ports):
            self.uic.cbPort.addItem(port)
            self.uic.cbPort_conveyor.addItem(port)
        self.ser[0].message.connect(self.Received_robot)
        self.ser[1].message.connect(self.Received_conveyor)

        # portss = QtSerialPort.QSerialPortInfo().availablePorts()

        self.test_lsv()

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
                self.timer_video.start(50)
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
                # print('pred : ', pred)
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
                            # clss.append(cls)
                            plot_one_box(xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=2)
                            # c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            # # print('c1: ', c1, 'c2: ', c2)
                            # center = (int((c1[0]+c2[0])/2), int((c1[1]+c2[1])/2))
                            # cv2.circle(showimg, center, radius=0, color=(0,0,255), thickness=3)

                        # Tracking ----***************
                        dets_to_sort = np.empty((0, 6))
                        # NOTE: We send in detected object class too
                        for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                            dets_to_sort = np.vstack((dets_to_sort,
                                                      np.array([x1, y1, x2, y2, conf, detclass])))

                        if self.track:
                            tracked_dets = self.sort_tracker.update(dets_to_sort, unique_color=False)
                            tracks = self.sort_tracker.getTrackers()

                            # draw boxes for visualization
                            if len(tracked_dets) > 0:
                                bbox_xyxy = tracked_dets[:, :4]
                                identities = tracked_dets[:, 8]
                                categories = tracked_dets[:, 4]
                                confidences = None

                                # print('bbox_xyxy=', bbox_xyxy)
                                # print('identities=', identities)
                                # print('categories=', categories)
                                # print('confidences=', confidences)
                                for i, box in enumerate(bbox_xyxy):
                                    x1, y1, x2, y2 = [int(i) for i in box]
                                    cat = int(categories[i]) if categories is not None else 0
                                    id = int(identities[i]) if identities is not None else 0

                                    center = (int((x1+x2)/2), int((y1+y2)/2))
                                    cv2.circle(showimg, center, radius=0, color=(0, 0, 255), thickness=3)   #draw center point
                                    tl = 2 or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
                                    tf = max(tl, 1)  # font thickness
                                    cv2.putText(showimg, 'ID:'+str(id), center, 0, tl/3, self.colors[cat],
                                                thickness=tf, lineType=cv2.LINE_AA)

                                # if opt.show_track:
                                #     # loop over tracks
                                #     for t, track in enumerate(tracks):
                                #         track_color = colors[int(track.detclass)] if not opt.unique_track_color else \
                                #         sort_tracker.color_list[t]
                                #
                                #         [cv2.line(im0, (int(track.centroidarr[i][0]),
                                #                         int(track.centroidarr[i][1])),
                                #                   (int(track.centroidarr[i + 1][0]),
                                #                    int(track.centroidarr[i + 1][1])),
                                #                   track_color, thickness=opt.thickness)
                                #          for i, _ in enumerate(track.centroidarr)
                                #          if i < len(track.centroidarr) - 1]
                        else:
                            bbox_xyxy = dets_to_sort[:, :4]
                            identities = None
                            categories = dets_to_sort[:, 5]
                            confidences = dets_to_sort[:, 4]


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
        if self.ser[0].is_Open():
            self.ClosePort()
        else:
            self.OpenPort()

    def OpenPort(self):
        portname = self.uic.cbPort.currentText()
        baud = int(self.uic.cbBaud.currentText())
        # parity = self.dictParity[self.uic.cbParity.currentIndex()]
        parity = self.uic.cbParity.currentIndex()
        self.ser[0].Conf(portName=portname, baudrate=baud, parity=parity)

        print("name:", self.ser[0].serialPort.portName(), "---baud:",
              self.ser[0].serialPort.baudRate(), "---Parity:", self.ser[0].serialPort.parity())

        self.ser[0].start()
        self.ser[0].Open()

    def ClosePort(self):
        self.ser[0].Close()
        self.ser[0].terminate()
        # self.ser[0].deleteLater()

    def btnConnect_conveyor_clicked(self):
        if self.ser[1].is_Open():
            self.ClosePort_conveyor()
        else:
            self.OpenPort_conveyor()
        self.count = 0
        self.pulse = 0
    def OpenPort_conveyor(self):
        portname = self.uic.cbPort_conveyor.currentText()
        baud = 9600
        parity = 0  #NONE
        # parity = self.uic.cbParity.currentIndex()
        self.ser[1].Conf(portName=portname, baudrate=baud, parity=parity)
        print("name:", self.ser[1].serialPort.portName(), "---baud:", self.ser[0].serialPort.baudRate(), "---Parity:",
              self.ser[1].serialPort.parity())

        self.ser[1].start()
        self.ser[1].Open()

        pid = [0x02, 0x53, 0x50, 0x49, 0x44, 0x00, 0x00, 0x00,
               0x05, 0x00, 0x00, 0x05, 0x01, 0x32, 0x02, 0x00, 0x16, 0x03]
        self.ser[1].sendSerial(bytes(pid))
    def ClosePort_conveyor(self
                           ):
        self.ser[1].Close()
        self.ser[1].terminate()

    def Timer_tick(self):
        if self.ser[0].is_Open():
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

        if self.ser[1].is_Open():
            self.uic.btnConnect_conveyor.setText('Close')
            self.uic.cbPort_conveyor.setDisabled(True)
        else:
            self.uic.btnConnect_conveyor.setText('Open')
            self.uic.cbPort_conveyor.setEnabled(True)

    def Received_robot(self, buff):
        print('rec robot=', buff)

    def Received_conveyor(self, buff):
        # print('buff=', buff)
        # print(len(buff))
        if len(buff) == 18 and buff[1:5]==b'SRUN':
            if buff[0]==2 and buff[17]==3:
                self.count+=1
                data = buff[8:16]
                direct = data[5]
                self.pulse += data[6]*256 + data[7]
                # print('pulse=', pulse)
                if self.count==10:
                    vel = int(self.pulse * 3600 / 4 / 11 / 56)   # độ trên giây
                    print(vel, 'dec/s')
                    self.uic.lb_realVel.setText(str(vel))
                    self.count = 0
                    self.pulse = 0


    def btnRun_conveyor_clicked(self):
        direct = 0x52   #R (REVERSE)
        direct = 0x46   #F (FORWARD)
        self.count = 0
        self.pulse = 0
        speed = self.uic.spinBox_Vel.value()
        sp0 = speed %256
        sp1 = speed //256

        runSpeed = [0x02, 0x53, 0x52, 0x55, 0x4E, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, direct, sp1, sp0, 0x16, 0x03]
        self.ser[1].sendSerial(bytes(runSpeed))

    def btnStop_conveyor_clicked(self):
        STOP = [0x02, 0x53, 0x54, 0x4F, 0x50, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x16, 0x03]
        self.ser[1].sendSerial(bytes(STOP))
        self.count = 0
        self.pulse = 0
        self.uic.lb_realVel.setText('0')

    def btnReset_robot_clicked(self):
        RESET = [0x02, 0x32, 0x03]  # 0x02 '2' 0x03
        self.ser[0].sendSerial(bytes(RESET))
    def btnStart_robot_clicked(self):
        START = [0x02, 0x30, 0x03]  # 0x02 '0' 0x03
        self.ser[0].sendSerial(bytes(START))
    def btnStop_robot_clicked(self):
        STOP = [0x02, 0x31, 0x03]   # 0x02 '1' 0x03
        self.ser[0].sendSerial(bytes(STOP))
    def test_lsv(self):
        lst = ["aaaa", 'bbb', 'ccc']
        listModel = QtCore.QStringListModel()
        listModel.setStringList(lst)
        self.uic.lsv_stack_object.setModel(listModel)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
