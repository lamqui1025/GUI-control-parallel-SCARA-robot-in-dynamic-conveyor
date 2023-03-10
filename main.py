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
from ScaleCam import ScaleCam
from controlScara import Scara

import torch
import torch.backends.cudnn as cudnn

import serial
import serial.tools.list_ports
import cv2
import numpy as np
import random
import queue
import math

class Objects:
    def __init__(self, id, cls, center, time):
        self.id = id
        self.cls = cls
        self.center = center
        self.time = time
        self.lstcens = []
        self.vel = None
        self.cnt_mistake = 0
        self.in_cats = True

class ModelYolov7:
    source = 0
    device = ''
    weights = 'best-1.pt'
    image_size = 640
    conf_thres = 0.8
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

class MainWindow(QMainWindow):
    dictParity = {0: 'N', 1: 'E', 2: 'O', 3: 'M'}
    dictCls = {0: 'USA', 1: 'Germany', 2: 'Vietnam', 3: 'UK',
               4: 'Brazil', 5: 'Canada', 6: 'Japan', 7: 'Malaysia'}

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
    # LIBRARY
    scam = ScaleCam()
    scara = Scara()

    #   FLAG    .....
    on_yolo = False
    is_scaleCam = False

    # argument
    dx_con = -288    # 288 mm
    dy_con = 210    # 205 mm

    #   properties
    current_id = 0
    STACK = queue.Queue()
    objs = []
    non_in_cats = 0
    numof_picked = [0, 0, 0, 0, 0, 0, 0, 0]
    def __init__(self):
        # self.main_win = QMainWindow()
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        # init for stream
        self.timer_video = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        # self.out = None

        # khai bao nut an chay
        self.uic.btnConnect.clicked.connect(self.btnConnect_clicked)
        self.uic.btnConnect_conveyor.clicked.connect(self.btnConnect_conveyor_clicked)
        # self.uic.btnStart.clicked.connect(self.start_capture_video)
        # self.uic.btnStop.clicked.connect(self.stop_capture_video)
        self.uic.btnRun_conveyor.clicked.connect(self.btnRun_conveyor_clicked)
        self.uic.btnStop_conveyor.clicked.connect(self.btnStop_conveyor_clicked)
        self.uic.btnReset_robot.clicked.connect(self.btnReset_robot_clicked)
        self.uic.btnStart_robot.clicked.connect(self.btnStart_robot_clicked)
        self.uic.btnStop_robot.clicked.connect(self.btnStop_robot_clicked)
        self.uic.btnReady_robot.clicked.connect(self.btnReady_robot_clicked)
        # self.uic.btnTest_robot.clicked.connect(self.btnTest_robot_clicked)
        self.uic.btnScaleCam.clicked.connect(self.btnScaleCam_clicked)
        self.uic.btnYolo_work.clicked.connect(self.btnYolo_clicked)
        self.uic.spinBox_vmax.valueChanged.connect(self.update_vmaxp)
        self.uic.spinBox_amax.valueChanged.connect(self.update_amaxp)

        self.timer_video.timeout.connect(self.show_video_frame)

        # Timer To Check all Status
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.Timer_tick)
        self.timer.start(200)
        # Timer To PICK object
        self.timer_pick_object = QtCore.QTimer()
        self.timer_pick_object.timeout.connect(self.pickObjects_timer_tick)
        self.timer_pick_object.start(20)
        # Create 2 Seirial
        self.ser = {}
        self.ser[0] = serialThread()
        self.ser[1] = serialThread()
        ports = serial.tools.list_ports.comports()
        for port, desc, hwid in sorted(ports):
            self.uic.cbPort.addItem(port)
            self.uic.cbPort_conveyor.addItem(port)
        self.ser[0].message.connect(self.Received_robot)
        self.ser[1].message.connect(self.Received_conveyor)
        # Emit from Scara
        # self.scara.respond.connect(self.scara_respond)
        ##  INITIALIZE mainWindow
        self.initialize_mainWindow()

    def initialize_mainWindow(self):
        self.start_capture_video()
        self.clsIcon = self.loadIcons()

        # update value in SpinBox vmaxp, amaxp (%) and vmax, amax (deg/s)
        self.uic.spinBox_vmax.setValue(self.scara.vMaxp)
        self.uic.spinBox_amax.setValue(self.scara.aMaxp)

        vmax = round(self.scara.vMax, 3)  # deg/s
        amax = round(self.scara.aMax, 3)  # deg/s
        self.uic.lb_vmax.setText(str(vmax))
        self.uic.lb_amax.setText(str(amax))

        # INIT TABLE WIDGETS
        self.uic.tb_clsObjs.setColumnCount(8)
        self.uic.tb_clsObjs.setRowCount(2)

        self.uic.tb_clsObjs.setRowHeight(0, 80)
        # print(self.uic.tb_clsObjs.width())
        for i in range(8):
            self.uic.tb_clsObjs.setColumnWidth(i, int(round(self.uic.tb_clsObjs.width()/8.5)))
            lb_cls = QtWidgets.QLabel()

            pixmap = QtGui.QPixmap('clsImg/' + self.dictCls[i] + '.png')
            pixmap = pixmap.scaled(80, 80, Qt.KeepAspectRatio)
            lb_cls.setPixmap(pixmap)
            lb_cls.resize(80, 80)
            self.uic.tb_clsObjs.setCellWidget(0, i, lb_cls)

        self.update_numof_picked()
        # self.test_lstwidget()

    def loadIcons(self):
        Icons = []
        for i in range(8):
            Icons.append(QtGui.QIcon('clsImg/' + self.dictCls[i] + '.png'))
        return Icons
    def update_vmaxp(self):
        vmaxp = self.uic.spinBox_vmax.value()
        self.scara.set_vMaxp(vmaxp)
        vmax = round(self.scara.vMax, 3)
        self.uic.lb_vmax.setText(str(vmax))
    def update_amaxp(self):
        amaxp = self.uic.spinBox_amax.value()
        self.scara.set_aMaxp(amaxp)
        amax = round(self.scara.aMax, 3)
        self.uic.lb_amax.setText(str(amax))

    # def keyPressEvent(self, event):
    #     if event.key() == Qt.Key.Key_Return:
    #         print('Press Enter!')
    #         # self.uic.spinBox_Vel.addAction()
    #         self.btnRun_conveyor_clicked()

    def closeEvent(self, event):
        self.stop_capture_video()

        self.timer_pick_object.stop()
        self.timer.stop()
        self.timer_video.stop()

        self.scara.terminate()
        self.ser[0].terminate()
        self.ser[1].terminate()

    def stop_capture_video(self):
        self.timer_video.stop()
        # self.uic.btnStart.setEnabled(True)
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
                # self.uic.btnStart.setDisabled(True)

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
    def btnYolo_clicked(self):
        if self.on_yolo:
            self.on_yolo = False
            self.uic.btnYolo_work.setText('START')
        else:
            self.is_scaleCam = False
            self.on_yolo = True
            self.uic.btnYolo_work.setText('STOP')
            self.uic.btnScaleCam.setText('Scale')
    def btnScaleCam_clicked(self):
        if self.is_scaleCam:
            self.is_scaleCam = False
            self.uic.btnScaleCam.setText('Scale')
        else:
            self.on_yolo = False
            self.is_scaleCam = True
            self.uic.btnScaleCam.setText('unScale')
            self.uic.btnYolo_work.setText('START')

    def show_video_frame(self):
        flag, img = self.cap.read()

        if img is not None:
            frame = img
            if self.on_yolo:
                time_detected = time_synchronized()
                frame, bboxs, identities, categories = self.detect_and_track(frame)
                if self.scam.scam_completed:
                    for i in range(len(identities)-1, -1, -1):
                        bbox_xyxy = bboxs[i]
                        id = identities[i]
                        cat = int(categories[i])
                        # print(bbox_xyxy, id, cat)
                        center = (bbox_xyxy[0:2] + bbox_xyxy[2:4])/2
                        # dH = self.scam.distance_line(center, self.pH)
                        # dW = self.scam.distance_line(center, self.pW)
                        # print('point:', center, dH*10/self.pp1cm, dW*10/self.pp1cm, 'Time:', time_detected)
                        # dx = round(dH*10/self.pp1cm + self.dx_con, 3)
                        # dy = round(dW*10/self.pp1cm + self.dy_con, 3)
                        # obj = Objects(id, cat, dx, dy, time_detected)
                        # if id > self.current_id:
                        #     self.current_id = id
                        #     # self.STACK.put(obj)
                        lim_top = self.P1[1] + self.pp1cm*1.1
                        lim_bot = self.P2[1] - self.pp1cm*1.1
                        if center[0]>100 and center[0]<560 and center[1]>lim_top and center[1]<lim_bot:
                            if id <= self.current_id:
                                for j in range(len(self.objs)):
                                    if self.objs[j].id == id and self.objs[j].cls == cat:
                                        pre_center = self.objs[j].center
                                        pre_time = self.objs[j].time
                                        self.objs[j].center = center
                                        self.objs[j].time = time_detected
                                        self.objs[j].lstcens.append(center)
                                        self.objs[j].vel = abs(pre_center - center)/(time_detected - pre_time)
                                    elif self.objs[j].id == id and self.objs[j].cls != cat:
                                        pre_center = self.objs[j].center
                                        pre_time = self.objs[j].time
                                        self.objs[j].center = center
                                        self.objs[j].time = time_detected
                                        self.objs[j].lstcens.append(center)
                                        self.objs[j].vel = abs(pre_center - center) / (time_detected - pre_time)

                                        self.objs[j].cnt_mistake += 1
                                        if self.objs[j].cnt_mistake > 1:
                                            self.objs[j].in_cats = False
                                            c1, c2 = (int(bbox_xyxy[0]), int(bbox_xyxy[1])), (int(bbox_xyxy[2]), int(bbox_xyxy[3]))
                                            cv2.rectangle(img, c1, c2, (255, 255, 255), thickness=2,
                                                          lineType=cv2.LINE_AA)
                            else:
                                if center[0] > 400:
                                    obj = Objects(id, cat, center, time_detected)
                                    obj.lstcens.append(center)
                                    self.objs.append(obj)
                                    self.current_id = id
                                else:
                                    c1, c2 = (int(bbox_xyxy[0]), int(bbox_xyxy[1])), (int(bbox_xyxy[2]), int(bbox_xyxy[3]))
                                    cv2.rectangle(img, c1, c2, (255,255,255), thickness=2, lineType=cv2.LINE_AA)
                        elif center[0]<=100 and center[1]>lim_top and center[1]<lim_bot:
                            objs = []
                            for j in range(len(self.objs)):
                                if self.objs[j].id == id:
                                    if self.objs[j].in_cats:
                                        if self.objs[j].cls == cat:
                                            # add obj in STACK
                                            self.STACK.put(self.objs[j])
                                            self.addTo_lswStack(self.objs[j])
                                            # del obj in list object
                                        else:
                                            self.non_in_cats += 1
                                    else:
                                        self.non_in_cats += 1
                                else:
                                    objs.append(self.objs[j])
                            self.objs = objs

            elif self.is_scaleCam:
                frame, P1, P2, self.pH, self.pW = self.scam.scaleCam(frame)
                self.scara.set_pH_pW(pH=self.pH, pW=self.pW)
                print('Scale Camera:', P1, P2, self.pH, self.pW)
                self.P1 = P1
                self.P2 = P2
                if self.scam.scam_completed:
                    dis12 = self.scam.distance(P1[0], P1[1], P2[0], P2[1])
                    self.pp1cm = dis12/12
                    self.scara.set_pp1cm(dis12/12)


            # for obj in self.objs:
            #     if not obj.in_cats:
            #         cx = int(obj.center[0])
            #         cy = int(obj.center[1])
            #         c1 = (cx - 60, cy - 60)
            #         c2 = (cx + 60, cy + 60)
            #         cv2.rectangle(frame, c1, c2, (255,255,255), thickness=2, lineType=cv2.LINE_AA)
            self.uic.lb_not_in_classes.setText(str(self.non_in_cats))
            frame[:, 560:561] = [0, 255, 0]
            frame[:, 639:640] = [0, 255, 0]
            frame[:, 100] = [0, 255, 0]
            self.result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            # showImage = showImage.scaled(700, 550, Qt.KeepAspectRatio)
            self.uic.lbScreen.setPixmap(QtGui.QPixmap.fromImage(showImage))

        else:
            self.timer_video.stop()
            self.cap.release()
            # self.out.release()
            self.uic.lbScreen.clear()
            # self.uic.btnStart.setDisabled(False)

    def detect_and_track(self, img):
        bbox_xyxy = []
        identities = []
        categories = []
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
                        # name_list.append(self.names[int(cls)])
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

                        # print(tracked_dets)
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

                                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                                cv2.circle(showimg, center, radius=0, color=(0, 0, 255),
                                           thickness=3)  # draw center point
                                tl = 2 or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
                                tf = max(tl, 1)  # font thickness
                                cv2.putText(showimg, 'ID:' + str(id), center, 0, tl / 3, self.colors[cat],
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

        show = showimg
        # show = cv2.resize(showimg, (640, 480))
        return show, bbox_xyxy, identities, categories

        # self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        # showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
        #                          QtGui.QImage.Format_RGB888)
        # # showImage = showImage.scaled(700, 550, Qt.KeepAspectRatio)
        # self.uic.lbScreen.setPixmap(QtGui.QPixmap.fromImage(showImage))

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
    def ClosePort_conveyor(self):
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

        if self.scara.is_running:
            if not self.scara.isRunning():
                self.scara.start()
        else:
            if self.scara.isRunning():
                self.scara.terminate()

    recdata=[]
    f_rec = False
    r_completed = False
    def Received_robot(self, buff):
        print('rec robot=', buff)
        for d in buff:
            if int(d)==2:
                self.f_rec = True
                self.r_completed = False
            elif int(d)==3:
                self.f_rec = False
                self.r_completed = True
            elif self.f_rec:
                self.recdata.append(d)

        if self.r_completed:
            self.r_completed = False
            data = self.recdata
            self.recdata = []
            # print('rec all data:', data)

            if data[0]==0:
                print('Thuc hien thanh cong!')
                self.scara.set_robot_running(True)   # Set robot is running
                if data[1]==1:
                    print('Positive Workspace!')
                else:
                    print('Negative Workspace!')

                realPul1 = self.fiveCharToValue(data[2:7])
                realPul4 = self.fiveCharToValue(data[7:12])

                self.scara.set_realPul1(realPul1)
                self.scara.set_realPul4(realPul4)

                self.uic.lb_vtheta1.setText(str(realPul1))
                self.uic.lb_vtheta4.setText(str(realPul4))
            elif data[0]==1:
                print('Phan giai yeu cau that bai!')
            elif data[0]==2:
                print('Thuc hien yeu cau that bai!')

            if self.scara.is_busy:
                if data[0]==0:
                    self.numof_picked[self.obj_picking.cls] += 1
                    self.update_numof_picked()
                elif data[0]==1 or data[0]==2:
                    self.obj_missed += 1
                    self.uic.lb_numof_missed.setText(str(self.obj_missed))
                self.scara.set_busy(False)  # Robot completed working

    def fiveCharToValue(self, fchar):
        sign = 1 if fchar[0]== ord('+') else -1
        value = 0
        for i in range(1, 5):
            value += (fchar[i] - ord('0')) * 10**(4-i)
        return sign*value

    def degToMmps(self, degps):
        return degps * 25 * np.pi / 360
    def mmToDegps(self, mmps):
        return mmps * 360 / (25 * np.pi)

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
                    vel = self.pulse * 10 * 25 * np.pi / 4 / 11 / 56   # ????? tr??n gi??y
                    print(vel, 'mm/s')
                    rvel = round(vel, 3)
                    self.velcon = vel
                    # self.scara.set_velcon_mmps(vel)
                    self.uic.lb_realVel.setText(str(rvel))
                    self.count = 0
                    self.pulse = 0


    def btnRun_conveyor_clicked(self):
        direct = 0x52   #R (REVERSE)
        direct = 0x46   #F (FORWARD)
        self.count = 0
        self.pulse = 0
        mmps = self.uic.spinBox_Vel.value()

        degps = int(round(self.mmToDegps(mmps)))    # deg/s
        mmpsx = self.degToMmps(degps)   # mm/s
        self.scara.set_velcon_mmps(mmpsx)   # mm/s

        sp0 = degps % 256
        sp1 = degps // 256

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
        self.scara.set_velcon_mmps(0)

    def btnReset_robot_clicked(self):
        RESET = [0x02, 0x32, 0x03]  # 0x02 '2' 0x03
        self.ser[0].sendSerial(bytes(RESET))
    def btnStart_robot_clicked(self):
        START = [0x02, 0x30, 0x03]  # 0x02 '0' 0x03
        self.ser[0].sendSerial(bytes(START))
    def btnStop_robot_clicked(self):
        STOP = [0x02, 0x31, 0x03]   # 0x02 '1' 0x03
        self.ser[0].sendSerial(bytes(STOP))
        self.scara.is_running = False
    def btnReady_robot_clicked(self):
        neg = 0x2D
        pos = 0x2B

        READY = [0x02, 0x37, 0x30, pos, 0x30, 0x30, 0x30, 0x30,
                 pos, 0x31, 0x30, 0x30, 0x30, 0x31, 0x30, 0x31, 0x30, 0x03]

        self.ser[0].sendSerial(bytes(READY))

    #   Check and pick object
    obj_picking = None
    obj_missed = 0
    sendBuff = None
    is_sending_robot = False
    timeToSend = 0
    def pickObjects_timer_tick(self):
        if self.scam.scam_completed and self.scara.is_running and self.scara.isRunning():
            if self.is_sending_robot:
                if time_synchronized() >= self.timeToSend:
                    if self.sendBuff is not None:
                        if self.ser[0].sendSerial(bytes(self.sendBuff)) != -1:
                            self.scara.set_busy(True)
                        self.is_sending_robot = False
            else:
                if not self.STACK.empty() and not self.scara.is_busy:
                    obj = self.STACK.get()
                    self.uic.lsw_stack_obj.takeItem(0)
                    self.obj_picking = obj
                    self.is_sending_robot = self.scara.pick_object(obj)
                    if self.is_sending_robot:
                        self.sendBuff = self.scara.get_sendBuff()
                        self.timeToSend = self.scara.get_timeToSend()
                    else:
                        self.obj_missed += 1
                        self.uic.lb_numof_missed.setText(str(self.obj_missed))

    def addTo_lswStack(self, obj):
        item = QtWidgets.QListWidgetItem()
        item.setText('ID: ' + str(obj.id) + '        ' + str(self.dictCls[obj.cls]))
        item.setIcon(self.clsIcon[obj.cls])
        self.uic.lsw_stack_obj.addItem(item)

    def update_numof_picked(self):
        for i in range(8):
            item = QtWidgets.QTableWidgetItem()
            item.setText(str(self.numof_picked[i]))
            font = QtGui.QFont()
            font.setBold(True)
            item.setFont(font)
            self.uic.tb_clsObjs.setItem(1, i, item)






    def test_lstwidget(self):
        obj = Objects(id=1, cls=0, center=[10, 10], time=10)
        self.addTo_lswStack(obj)
        obj = Objects(id=2, cls=1, center=[10, 10], time=10)
        self.addTo_lswStack(obj)

        item = self.uic.lsw_stack_obj.item(0)
        item.setText('Edited!')


    # def test_lsv(self):
    #     lst = ["aaaa", 'bbb', 'ccc']
    #     listModel = QtCore.QStringListModel()
    #     listModel.setStringList(lst)
    #     self.uic.lsv_stack_object.setModel(listModel)
    #     self.uic.lsv_stack_object.

    # def valueTolst(self, value):
    #     sign = '+' if value>=0 else '-'
    #     val = abs(int(value*10))
    #     sval = ''
    #     for i in range(4):
    #         sval = str(val%10) + sval
    #         val = val//10
    #     sval = sign + sval
    #     lstValue = [ord(c) for c in sval]
    #     return lstValue

    def btnTest_robot_clicked(self):
        neg = 0x2D
        pos = 0x2B
        # TEST = [0x02, 0x39, 0x30, neg, 0x31, 0x30, 0x30, 0x30, pos, 0x32, 0x35, 0x30, 0x30,
        #         pos, 0x31, 0x30, 0x30, 0x30, pos, 0x32, 0x35, 0x30, 0x30, 0x31, 0x35, 0x31, 0x35, 0x03]

        print('STACK.empty()=', self.STACK.empty())
        if not self.STACK.empty():
            obj = self.STACK.get()
            id = obj.id
            cls = obj.cls
            dx = obj.dx
            dy = obj.dy

            pointAx = self.valueTolst(dx)
            pointAy = self.valueTolst(dy)

            pointBx = [pos, 0x31, 0x30, 0x30, 0x30]
            pointBy = [pos, 0x32, 0x35, 0x30, 0x30]
            STX = [0x02]
            CMD = [0x39, 0x30]      # '9', '0'
            VEL = [0x31, 0x35]      # '1', '5'
            ACC = [0x31, 0x35]      # '1', '5'
            ETX = [0x03]

            sendBuff = STX + CMD + pointAx + pointAy + pointBx + pointBy + VEL + ACC + ETX

            self.ser[0].sendSerial(bytes(sendBuff))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
