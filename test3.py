import time

import numpy as np
import math
from utils.torch_utils import time_synchronized

_longLink = 225  # mm
_shortLink = 150
_disBase = 150

_microStep =  32  #step of stepper motor
_degPerStep = 1.8/_microStep
_stepPerDeg = _microStep/1.8
_vmax = 1080 #deg/second
_amax = 3*_vmax #deg/second^2

vMax = 17 * _vmax/100  # % * _vmax (deg/second)
aMax = 12 * _amax/100  # % * _amax (deg/second^2)

def calVmax(s_max, a_max, v_max):
    return min(min(float(np.sqrt(abs(s_max * a_max))), v_max), _vmax)

def pointToAngle1(x, y):
    theta1 = [0, 0]
    # Be1, Ce1, Ae1, De11, theta11, Be4, Ce4, Ae4, De41, theta41;
    a1 = _disBase
    a2 = _shortLink
    a3 = _longLink
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

def cal_tf(pointA, pointB, vmax, amax):
    #   tf to pointA
    thetaA = pointToAngle1(pointA[0], pointA[1])
    # s1 = thetaA[0] - _realPul1 *_degPerStep
    # s4 = thetaA[1] - _realPul4 *_degPerStep
    # vmax1 = calVmax(s1, amax, vmax)
    # vmax4 = calVmax(s4, amax, vmax)
    # tf1 = max(abs(s1 / vmax1) + vmax1 / amax, abs(s4 / vmax4) + vmax4 / amax)
    #   tf to pointB
    thetaB = pointToAngle1(pointB[0], pointB[1])
    s1 = thetaB[0] - thetaA[0]
    s4 = thetaB[1] - thetaA[1]
    vmax1 = calVmax(s1, amax, vmax)
    vmax4 = calVmax(s4, amax, vmax)
    tf2 = max(abs(s1 / vmax1) + vmax1 / amax, abs(s4 / vmax4) + vmax4 / amax)
    return tf2

tf = cal_tf([200, 70], [-150, 280], vMax, aMax)
print(tf + 20/1000 + 200/1000)
# while(True):
#     time.sleep(0.1)
#     print(time_synchronized())