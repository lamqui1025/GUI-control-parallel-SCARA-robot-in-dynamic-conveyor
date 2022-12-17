import cv2
import numpy as np

class ScaleCam:
    P1 = None
    P2 = None
    pA = None
    pB = None
    pH = None
    pW = None
    scam_completed = False

    def distance_line(self, point, line):
        [x, y] = point
        [a, b] = line
        return abs(a*x - y + b)/np.sqrt(1+a**2)
    def distance(self, a1, a2, b1, b2):
        return np.sqrt((a1 - b1) ** 2 + (a2 - b2) ** 2)

    def slove2(self, a1, a2, b1, b2, c1, c2):
        D = a1 * b2 - a2 * b1
        Dx = c1 * b2 - c2 * b1
        Dy = a1 * c2 - a2 * c1
        x = Dx / D
        y = Dy / D
        return x, y

    def perpendicular_line(self, a, x, y):
        pa = -1 / a
        pb = y - pa * x
        return pa, pb

    def scaleCam(self, frame):
        self.scam_completed = False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        edgesc = cv2.Canny(blur, 60, 100, 15)
        # cv2.imshow('edgesc', edgesc)
        imgcp2 = frame.copy()
        rows = edgesc.shape[0]
        circles = cv2.HoughCircles(edgesc, cv2.HOUGH_GRADIENT, 1, rows / 4, param1=45, param2=15, minRadius=10,
                                           maxRadius=20)
        # print('circles=', circles)
        if isinstance(circles, np.ndarray):
            # circles = np.uint16(np.around(circles))
            redPoints = []
            for cir in circles[0, :]:
                if cir[0] > 560 and cir[0] < 640:
                    redPoints.append(cir)
                    x, y, r = np.array(cir, dtype='uint16')
                    cv2.circle(imgcp2, (x, y), r, (0, 255, 0), 3)
            if len(redPoints) == 2:
                p1 = redPoints[0][0:2]
                p2 = redPoints[1][0:2]
                if self.P1 is not None and self.P2 is not None:
                    if p1[1] < p2[1]:
                        self.P1 = (self.P1 + p1) / 2
                        self.P2 = (self.P2 + p2) / 2
                    else:
                        self.P1 = (self.P1 + p2) / 2
                        self.P2 = (self.P2 + p1) / 2
                else:
                    if p1[1] < p2[1]:
                        self.P1 = p1
                        self.P2 = p2
                    else:
                        self.P1 = p2
                        self.P2 = p1
                # print('P1=', self.P1, 'P2=', self.P2)
                dist = self.distance(self.P1[0], self.P1[1], self.P2[0], self.P2[1])
                # print('distance=', dist)
                a, b = self.slove2(self.P1[0], self.P2[0], 1, 1, self.P1[1], self.P2[1])
                self.pH = [a, b]
                # print(f'y={a}*x + {b}')
                # cenPoint = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))

                paA, pbA = self.perpendicular_line(a, self.P1[0], self.P1[1] + dist / 12)
                self.pA = [paA, pbA]
                self.pW = self.pA
                # print('pa=', paA, 'pb=', pbA)
                ppA1 = (0, int(paA * 0 + pbA))
                ppA2 = (640, int(paA * 640 + pbA))
                # print(ppA1, ppA2)
                imgcp2 = cv2.line(imgcp2, ppA1, ppA2, color=(0, 255, 255), thickness=1)

                paB, pbB = self.perpendicular_line(a, self.P2[0], self.P2[1] - dist / 12)
                self.pB = [paB, pbB]
                # print('pa=', paB, 'pb=', pbB)
                ppB1 = (0, int(paB * 0 + pbB))
                ppB2 = (640, int(paB * 640 + pbB))
                # print(ppB1, ppB2)
                imgcp2 = cv2.line(imgcp2, ppB1, ppB2, color=(0, 255, 255), thickness=1)

                self.scam_completed = True
                # cv2.circle(imgcp2, cenPoint, radius=0, color=(0, 255, 255), thickness=3)  # draw center point
        # print('cenPoint=', cenPoint)
        # imgcp2[:, 560:561] = [0, 255, 0]
        # imgcp2[:, 639:640] = [0, 255, 0]
        return imgcp2, self.P1, self.P2, self.pH, self.pW