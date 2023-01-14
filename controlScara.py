# import serial
from PyQt5.QtCore import pyqtSignal, QThread, QObject
# from PyQt5 import QtSerialPort, QtCore
import time
import numpy as np
import math
from utils.torch_utils import time_synchronized
from sklearn import datasets, linear_model

class Scara(QThread):
    dictDes = {0: [195, 70], 1: [195, 150],
               2: [135, 150], 3: [75, 150],
               4: [15, 150], 5: [-45, 150],
               6: [-105, 150], 7: [-165, 150]}

    #Flag Status of paralell SCARA
    is_running = False
    is_busy = False

    dx_con = -285  # -290 mm # -300 mm
    dy_con = 210  # 205 mm
    pp1cm = None
    pH = None
    pW = None
    velcon_mmps = 0   # mm/s
    D_conveyor = 25  # đường kính 25mm

    #   significant parallel Scara robot
    _longLink = 225  # mm
    _shortLink = 150
    _disBase = 150

    _realPul1 = 3200
    _realPul4 = 0
    _realX = 0
    _realY = 0

    _microStep = 32  #step of stepper motor
    _degPerStep = 1.8/_microStep
    _stepPerDeg = _microStep/1.8
    _vmax = 1080 #deg/second
    _amax = 3*_vmax #deg/second^2

    vMaxp = 30  # %
    aMaxp = 27  # %

    vMax = vMaxp * _vmax/100  # deg/second
    aMax = aMaxp * _amax/100  # deg/second

    timeToSend = 0
    sendBuff = None

    # respond = pyqtSignal(bool)
    def __init__(self, parent=None):
        super(Scara, self).__init__(parent)
        self.start()

    def set_robot_running(self, is_running):
        self.is_running = is_running

    def set_busy(self, busy):
        self.is_busy = busy

    def set_sending(self, sending):
        self.is_sending = sending

    def get_sendBuff(self):
        return self.sendBuff
    def get_timeToSend(self):
        return self.timeToSend

    def set_pH_pW(self, pH=None, pW=None):
        self.pH = pH
        self.pW = pW

    def set_pp1cm(self, pp1cm):
        self.pp1cm = pp1cm

    def set_velcon_mmps(self, vel_mmps):
        self.velcon_mmps = vel_mmps

    def set_realPul1(self, realPul1):
        self._realPul1 = realPul1

    def set_realPul4(self, realPul4):
        self._realPul4 = realPul4

    def distance_line(self, point, line):
        [x, y] = point
        [a, b] = line
        return abs(a*x - y + b)/np.sqrt(1+a**2)

    def set_vMaxp(self, vmaxp):
        self.vMaxp = vmaxp
        self.vMax = vmaxp * self._vmax / 100

    def set_aMaxp(self, amaxp):
        self.aMaxp = amaxp
        self.aMax = amaxp * self._amax / 100

    def valueTolst(self, value):
        sign = '+' if value>=0 else '-'
        val = abs(int(value*10))
        sval = ''
        for i in range(4):
            sval = str(val%10) + sval
            val = val//10
        sval = sign + sval
        lstValue = [ord(c) for c in sval]
        return lstValue

    def pixelToXy(self, points):
        XYs = []
        for poi in points:
            dH = self.distance_line(poi, self.pH)
            dW = self.distance_line(poi, self.pW)

            dx = round(dH*10/self.pp1cm + self.dx_con, 3)
            dy = round(dW*10/self.pp1cm + self.dy_con, 3)

            XYs.append([dx, dy])
        return XYs

    def calVmax(self, s_max, a_max, v_max):
        return min(min(float(np.sqrt(abs(s_max * a_max))), v_max), self._vmax)

    def pointToAngle1(self, x, y):
        theta1 = [0, 0]
        # Be1, Ce1, Ae1, De11, theta11, Be4, Ce4, Ae4, De41, theta41;
        a1 = self._disBase
        a2 = self._shortLink
        a3 = self._longLink
        # caculate theta1
        Be1 = (pow((x + a1 / 2), 2) + pow(y, 2) + pow(a2, 2) - pow(a3, 2)) / (2 * a2)
        Ce1 = -(x + a1 / 2)
        Ae1 = y
        De11 = (Ae1 + np.sqrt(pow(Ae1, 2) - pow(Be1, 2) + pow(Ce1, 2))) / (Be1 - Ce1)
        theta11 = 2 * math.atan(De11)
        if theta11 is None:
            theta11 = math.pi

        theta1[0] = theta11 * 180 / math.pi
        if theta1[0] < 0:
            theta1[0] = theta1[0] + 360
        Be4 = (pow((x - a1 / 2), 2) + pow(y, 2) + pow(a2, 2) - pow(a3, 2)) / (2 * a2)
        Ce4 = -(x - a1 / 2)
        Ae4 = y
        De41 = (Ae4 - np.sqrt(pow(Ae4, 2) - pow(Be4, 2) + pow(Ce4, 2))) / (Be4 - Ce4)
        theta41 = 2 * math.atan(De41)
        theta1[1] = theta41 * 180 / math.pi
        return theta1

    def cal_tf(self, pointA, pointB, vmax, amax):
        #   tf to pointA
        thetaA = self.pointToAngle1(pointA[0], pointA[1])
        s1 = thetaA[0] - self._realPul1*self._degPerStep
        s4 = thetaA[1] - self._realPul4*self._degPerStep
        vmax1 = self.calVmax(s1, amax, vmax)
        vmax4 = self.calVmax(s4, amax, vmax)
        tf1 = max(abs(s1 / vmax1) + vmax1 / amax, abs(s4 / vmax4) + vmax4 / amax)
        #   tf to pointB
        # thetaB = self.pointToAngle1(pointB[0], pointB[1])
        # s1 = thetaB[0] - thetaA[0]
        # s4 = thetaB[1] - thetaA[1]
        # vmax1 = self.calVmax(s1, amax, vmax)
        # vmax4 = self.calVmax(s4, amax, vmax)
        # tf2 = max(abs(s1 / vmax1) + vmax1 / amax, abs(s4 / vmax4) + vmax4 / amax)
        return tf1

    def pick_object(self, obj):
        id = obj.id
        cls = obj.cls
        center = obj.center   # pixel
        time_exist = obj.time
        vel = obj.vel   # mm/s
        lstcens = obj.lstcens

        #   LINEAR REGRESSION
        XYs = np.array(self.pixelToXy(lstcens))
        centers = np.array(lstcens)
        X = np.array([XYs[:, 0]]).T
        y = np.array([XYs[:, 1]]).T
        one = np.ones((X.shape[0], 1))
        Xbar = np.concatenate((one, X), axis=1)
        regr = linear_model.LinearRegression(fit_intercept=False)
        regr.fit(Xbar, y)

        W = regr.coef_[0]  #   W of line to predict object
        # print('W=', W)

        # velocity of conveyor
        mmps_con = self.velcon_mmps   # mm/s

        des_of_cls = self.dictDes[cls]     # destination for cls of object

        dH = self.distance_line(center, self.pH)
        dW = self.distance_line(center, self.pW)

        dx = round(dH * 10 / self.pp1cm + self.dx_con, 3)   # mm
        dy = round(dW * 10 / self.pp1cm + self.dy_con, 3)   # mm

        at_predtime = time_synchronized()
        pred_dx = dx + (at_predtime - time_exist)*mmps_con  # du doan vi tri hien tai cua vat the
        pred_dy = pred_dx*W[1] + W[0]
        # print('pred_dx=',  pred_dx, '----pred_dy=', pred_dy)
        tf_pred = self.cal_tf([pred_dx, pred_dy], des_of_cls, self.vMax, self.aMax) * 1.2 + 0.1  # them du sai so 25%
        if tf_pred is None:
            print('Fail to cal tf_pre')
            return False
        print('tf_pred=', tf_pred)
        x_delay = mmps_con * tf_pred    # mm
        print('x_delay=', x_delay)
        pick_dx = pred_dx + x_delay  # Gap vat truoc du doan 50mm
        if pick_dx > 180 - x_delay:
            print('Fail To PICK this Object!')
            self.is_sending = False
            return False

        pick_dy = pick_dx*W[1] + W[0]
        tf = self.cal_tf([pick_dx, pick_dy], des_of_cls, self.vMax, self.aMax)
        tf = tf if tf*1000 >= 60 else 40/1000   # New thoi gian qua ngan thi dung rawMoveToPoint() voi tong 60ms
        tf += 20/1000 + 200/1000 + 0.03  # Cong them 1 chu ki bu sau cap xung cho 2 dong co
        print('tf=', tf)

        # caculate time to pick_destination
        s = pick_dx - dx
        time_to_pick = s / mmps_con + time_exist
        time_to_send = time_to_pick - tf
        # print('time to Pick:', time_to_pick)
        # print('time to Send:', time_to_send)
        # print('time_synchronized:', time_synchronized())
        # print('PICK:', pick_dx, pick_dy)

        #   Prepare Buffer to Send
        STX = [0x02]
        CMD = [0x39, 0x30]  # '9', '0'
        pointAx = self.valueTolst(pick_dx)
        pointAy = self.valueTolst(pick_dy)
        pointBx = self.valueTolst(des_of_cls[0])
        pointBy = self.valueTolst(des_of_cls[1])
        VEL = [self.vMaxp//10 + ord('0'), self.vMaxp % 10 + ord('0')]    # %
        ACC = [self.aMaxp//10 + ord('0'), self.aMaxp % 10 + ord('0')]    # %
        ETX = [0x03]

        self.sendBuff = STX + CMD + pointAx + pointAy + pointBx + pointBy + VEL + ACC + ETX
        #   Waiting to SEND
        self.is_sending = True
        self.timeToSend = time_to_send
        return True




        # # du phong, bo di
        # present_time = time_synchronized()
        #
        # predict_cenx = (tf + time_synchronized() - time_exist) * mmps_con * self.pp1cm/10
        # predict_ceny = predict_cenx*W[1] + W[0]
        # predict_center = [predict_cenx, predict_ceny]
        # dH = self.distance_line(predict_center, self.pH)
        # dW = self.distance_line(predict_center, self.pW)
        #
        # dx = round(dH*10/self.pp1cm + self.dx_con, 3)
        # dy = round(dW*10/self.pp1cm + self.dy_con, 3)
        #
        # #   SEND SERIAL TO ROBOT
        # pointAx = self.valueTolst(dx)
        # pointAy = self.valueTolst(dy)
        #
        # neg = 0x2D
        # pos = 0x2B
        #
        # pointBx = [pos, 0x31, 0x30, 0x30, 0x30]
        # pointBy = [pos, 0x32, 0x35, 0x30, 0x30]
        # STX = [0x02]
        # CMD = [0x39, 0x30]  # '9', '0'
        # VEL = [0x31, 0x35]  # '1', '5'
        # ACC = [0x31, 0x35]  # '1', '5'
        # ETX = [0x03]
        #
        # sendBuff = STX + CMD + pointAx + pointAy + pointBx + pointBy + VEL + ACC + ETX
        #
        # ser.sendSerial(bytes(sendBuff))
