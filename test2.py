import types

import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture()
flag = cap.open(1)
cenPoint = ()
P1=None
P2=None

def distance(a1, a2, b1, b2):
  return np.sqrt((a1-b1)**2 + (a2-b2)**2)

def slove2(a1, a2, b1, b2, c1,c2):
    D = a1*b2-a2*b1
    print('D=', D)
    Dx = c1*b2 - c2*b1
    Dy = a1*c2 - a2*c1
    x = Dx/D
    y = Dy/D
    return x,y
def perpendicular_line(a, x, y):
    pa = -1/a
    pb = y - pa*x
    return pa, pb
while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11,11), 0)
        edgesc = cv2.Canny(blur, 60, 100, 15)
        cv2.imshow('edgesc', edgesc)
        imgcp2 = frame.copy()
        rows = edgesc.shape[0]
        circles = cv2.HoughCircles(edgesc, cv2.HOUGH_GRADIENT, 1, rows / 4, param1=45, param2=15, minRadius=10,
                                   maxRadius=20)
        print('circles=', circles)
        if isinstance(circles, np.ndarray):
            # circles = np.uint16(np.around(circles))
            redPoints=[]
            for cir in circles[0, :]:
                if cir[0]>560 and cir[0]<640:
                    redPoints.append(cir)
                    x, y, r = np.array(cir, dtype='uint16')
                    cv2.circle(imgcp2, (x, y), r, (0,255,0), 3)
            if len(redPoints) == 2:
                p1 = redPoints[0][0:2]
                p2 = redPoints[1][0:2]
                if P1 is not None and P2 is not None:
                    if p1[1]<p2[1]:
                        P1 = (P1+p1)/2
                        P2 = (P2+p2)/2
                    else:
                        P1 = (P1+p2)/2
                        P2 = (P2+p1)/2
                else:
                    if p1[1] < p2[1]:
                        P1 = p1
                        P2 = p2
                    else:
                        P1 = p2
                        P2 = p1
                print('p1=', p1,'p2=', p2)
                dist = distance(P1[0], P1[1], P2[0], P2[1])
                print('distance=', dist)
                a, b = slove2(P1[0], P2[0], 1, 1, P1[1], P2[1])
                print(f'y={a}*x + {b}')
                cenPoint = (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))

                paA, pbA = perpendicular_line(a, P1[0], P1[1]+dist/12)
                print('pa=', paA, 'pb=', pbA)
                ppA1 = (0, int(paA*100+pbA))
                ppA2 = (1200, int(paA*600+pbA))
                print(ppA1, ppA2)
                imgcp2 = cv2.line(imgcp2, ppA1, ppA2, color=(0, 255, 255), thickness=1)

                paB, pbB = perpendicular_line(a, P2[0], P2[1] - dist / 12)
                print('pa=', paB, 'pb=', pbB)
                ppB1 = (0, int(paB * 100 + pbB))
                ppB2 = (1200, int(paB * 600 + pbB))
                print(ppB1, ppB2)
                imgcp2 = cv2.line(imgcp2, ppB1, ppB2, color=(0, 255, 255), thickness=1)
                cv2.circle(imgcp2, cenPoint, radius=0, color=(0, 255, 255), thickness=3)  # draw center point
        print('cenPoint=', cenPoint)
        imgcp2[:, 560:561] = [0, 255, 0]
        imgcp2[:, 639:640] = [0, 255, 0]
        cv2.imshow('CAM1', imgcp2)
        print('shape=', imgcp2.shape)
        key = cv2.waitKey(30)
        print('-------------------------------------------')
        if key==ord('q'):
            break
    else:
        break

