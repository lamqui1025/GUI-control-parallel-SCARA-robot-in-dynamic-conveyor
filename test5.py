import cv2
import numpy as np
import matplotlib.pyplot as plt



def preprocessing(img):
  blur = cv2.GaussianBlur(img, (3, 3), 0)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
  morph_image = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
  sub_morp_image = cv2.subtract(blur,morph_image)
  canny = cv2.Canny(blur, 27, 255)
  ret, thresh = cv2.threshold(canny, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
  cv2.imshow('dilate', dilate)
  return dilate


def detect_plate(img, pre_img):
    cnts, _ = cv2.findContours(pre_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    screenCnt = None
    for c in cnts:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        screenCnt = None
        if len(approx) == 4:
            if (abs((approx[0] - approx[1]).any()) <= 20 and abs((approx[1] - approx[2]).any()) <= 20):
                screenCnt = approx
                break

    # Masking the part other than the number plate
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Now crop
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped = new_image[topx - 5:bottomx + 5, topy - 5:bottomy + 5]

    cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 1)
    cv2.imshow(img)
    return cropped

# def apply_and_show(image, filter):
#   #FFT
#     fft = np.fft.fft2(image)
#     fftshift = np.fft.fftshift(fft)
#
#     #Apply Filter
#     fftshift_filter = fftshift * filter
#
#     #IFFT
#     ifftshift = np.fft.ifftshift(fftshift_filter)
#     image2 = np.abs(np.fft.ifft2(ifftshift))
#
#     visualization_oneline([image, log_spec(fftshift), filter], ['Image', 'Spectrum', 'Filter'])
#     visualization_oneline([log_spec(fftshift_filter), log_spec(ifftshift), image2],['FFTshift_filter', 'IFFTShift','Filtered image'])
#     return fftshift, fftshift_filter

def Visualization(src, name, histogram = False, CDF = False, spectrum = False, figsz = None):
  num_x = 1
  if histogram: num_x+=1
  if spectrum: num_x+=1
  if figsz != None:
    fig, axes = plt.subplots(num_x, len(src), figsize = (figsz[0], figsz[1]))
  else:
    fig, axes = plt.subplots(num_x, len(src), figsize = (5*len(src), 5*num_x))
  for i in range(len(src)):
    j = 0
    axes[j][i].imshow(src[i], cmap='gray', vmin=0, vmax=255)
    if histogram:
      j += 1
      axes[j][i].hist(src[i].ravel(), 256, [0,256], label = 'PDF')
      if CDF:
        ax2 = axes[j][i].twinx()
        ax2.hist(src[i].ravel(), 256, [0,256], cumulative = True, histtype = 'step', color = 'red', label = 'CDF')
      axes[j][i].set_title("Histogram - CDF")

    if spectrum:
      j+=1
      f = np.fft.fft2(src[i])
      fshift = np.fft.fftshift(f)
      magnitude_spectrum = 20*np.log(np.abs(fshift))
      axes[j][i].imshow(magnitude_spectrum, cmap = 'gray')
    j = 0
    axes[j][i].set_title(name[i])
    if histogram:
      j += 1
      axes[j][i].set_title('Histogram ' + name[i])
    if spectrum:
      j += 1
      axes[j][i].set_title('Spectrum ' + name[i])
  fig.tight_layout()

def visualization_oneline(*args):
  fig, axes = plt.subplots(1, len(args[0]), figsize=(5*len(args[0]), 8), constrained_layout = False)
  for idx, img in enumerate(args[0]):
    axes[idx].imshow(img, cmap='gray')
    if len(args) > 1:
      axes[idx].set_title(args[1][idx])
  fig.tight_layout

def log_spec(spectrum):
  return 20 * np.log(1 + np.abs(spectrum))

def distance(a1, a2, b1, b2):
  return np.sqrt((a1-b1)**2 + (a2-b2)**2)

def apply_and_show(image, filter):
  #FFT
  fft = np.fft.fft2(image)
  fftshift = np.fft.fftshift(fft)

  #Apply Filter
  fftshift_filter = fftshift * filter

  #IFFT
  ifftshift = np.fft.ifftshift(fftshift_filter)
  image2 = np.abs(np.fft.ifft2(ifftshift))

  visualization_oneline([image, log_spec(fftshift), filter], ['Image', 'Spectrum', 'Filter'])
  visualization_oneline([log_spec(fftshift_filter), log_spec(ifftshift), image2], ['FFTshift_filter', 'IFFTShift','Filtered image'])
  return fftshift, fftshift_filter

#   --------------------------------------------
cap = cv2.VideoCapture()
flag = cap.open('obj_run.mp4')
backSub = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if ret:
        img = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equa = cv2.equalizeHist(gray)
        cv2.imshow('EQUALIZATION', equa)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = equa

        gamma = 1.5
        c = 255 / (np.amax(gray) ** gamma)
        dst2 = c * (gray ** gamma)
        dst2 = np.array(dst2, dtype='uint8')

        cv2.imshow('DST2', gray)

        fgMask = backSub.apply(img)
        fgMask = cv2.cvtColor(fgMask, 0)

        cv2.imshow('fgMask', fgMask)

        fgMask = cv2.GaussianBlur(fgMask, (5, 5), 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)



        # kernel = np.ones((9, 9), np.uint8)
        # fgMask = cv2.dilate(fgMask, kernel, iterations=1)
        #
        # kernel = np.ones((15, 15), np.uint8)
        # fgMask = cv2.erode(fgMask, kernel, iterations=1)
        #
        # kernel = np.ones((5, 5), np.uint8)
        # fgMask = cv2.dilate(fgMask, kernel, iterations=1)

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        # fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)

        _, fgMask = cv2.threshold(fgMask, 130, 255, cv2.THRESH_BINARY)

        cv2.imshow('FGMASK 2', fgMask)

        # dst2 = cv2.medianBlur(dst2, 5)

        # hpf = gray - cv2.GaussianBlur(equa, (21, 21), 3) + 127

        # cv2.imshow('HPF', hpf)

        # fft = np.fft.fft2(blur)
        # fftshift = np.fft.fftshift(fft)
        # fftshift = np.abs(fftshift)

        # rows = fftshift.shape[0]
        # cols = fftshift.shape[1]
        # D0 = 100
        # n = 10
        # crow = rows/2
        # ccol = cols/2

        # crow1 = fftshift.shape[0] // 2
        # ccol1 = fftshift.shape[1] // 2
        # size = 3
        # filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        # filter = np.pad(filter,
        #                 ((crow1 - size // 2, crow1 - size // 2 - 1), (ccol1 - size // 2, ccol1 - size // 2 - 1)),
        #                 'constant')
        # fft_filter = np.fft.fft2(filter)
        # filter_ffts = np.abs(np.fft.fftshift(fft))

        # D0 = 50
        # filter = np.ones((rows, cols))
        # for i in range(rows):
        #     for j in range(cols):
        #         filter[i, j] = 1 - np.exp(-distance(i, j, crow, ccol) ** 2 / (2 * (D0 ** 2)))

        # filter = np.zeros((rows, cols))
        # for i in range(rows):
        #     for j in range(cols):
        #         filter[i, j] = 1 / (1 + pow(distance(i, j, crow, ccol) / D0, 2 * n))

        # fftshift_filter = fftshift*filter
        #
        # ffts = np.uint8(log_spec(fftshift_filter))
        # cv2.imshow('ffts', ffts)

        # IFFT
        # ifftshift = np.fft.ifftshift(fftshift_filter)
        # image2 = np.abs(np.fft.ifft2(ifftshift))
        #
        # cv2.imshow('img2', image2)

        # visualization_oneline([equa, log_spec(fftshift)], 'equalize', 'fftshift')
        # pre_img = preprocessing(equa)

        # obj = detect_plate(frame, pre_img)
        # print(obj)

        # dst2 = cv2.GaussianBlur(dst2, (5, 5), 0)
        #
        # # ret, thresh = cv2.threshold(dst2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # thresh = cv2.adaptiveThreshold(dst2, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        #                      cv2.THRESH_BINARY_INV, 11, 2)
        # cv2.imshow('Threshold1', thresh)
        #
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # cv2.imshow('Threshold2', thresh)
        #


        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # # thresh = cv2.morphologyEx(thresh, cv2.MORPH_, kernel)
        #
        # thresh = cv2.dilate(thresh, kernel, iterations=1)
        # cv2.imshow('Threshold3', thresh)
        #
        # thresh = cv2.erode(thresh, kernel, iterations=1)
        # cv2.imshow('Threshold4', thresh)

        # contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print('contour.len=', len(contours))
        #
        # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        # cv2.imshow('Contours', img)



        # blur = cv2.GaussianBlur(thresh, (5, 5), 0)
        # edgesc = cv2.Canny(thresh, 50, 255)
        #
        # cv2.imshow('edgesc', edgesc)
        # imgcp2 = frame.copy()
        # rows = frame.shape[0]
        # circles = cv2.HoughCircles(edgesc, cv2.HOUGH_GRADIENT, 1, rows / 4, param1=100, param2=30, minRadius=35,
        #                            maxRadius=60)

        # print(circles)
        # if circles is not None:
        #     for cir in circles[0, :]:
        #         x, y, r = np.array(cir, dtype='uint16')
        #         cv2.circle(imgcp2, (x, y), r, (0, 255, 0), 3)
        # cv2.imshow('CAM1', imgcp2)
        # print('shape=', imgcp2.shape)

        key = cv2.waitKey(50)
        print('-------------------------------------------')
        if key == ord('q'):
            break
    else:
        break