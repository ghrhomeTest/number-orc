import cv2
import numpy as np
from matplotlib import pyplot as plt


# 寻找圆形小数点的位置 返回小数点横坐标和缩放比例
def circle_position(re):
    imgcir = cv2.imread(re)
    imgcirheight = imgcir.shape[0]
    imgcirgray = cv2.cvtColor(imgcir, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(imgcirgray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=5, maxRadius=50)
    beishu = imgcirheight / float(90)
    deci = 0
    if circles is not None:
        for circle in circles[0]:
            if circle[1] > (imgcirheight / 3 * 2):
                deci = int(circle[0])
    else:
        deci = 0

    return deci, beishu


# 拍个照
class CaptureVideo:
    @staticmethod
    def cap_video():
        # 定义摄像头
        capture = cv2.VideoCapture(0)
        error_str = " IO Error"
        if capture.isOpened() is False:
            raise error_str
        capture.set(3, 1920)
        capture.set(4, 1080)
        alpha = float(2.5)
        # cv2.namedWindow("Capture", cv2.WINDOW_NORMAL)

        while capture.isOpened():

            ret, image = capture.read()

            if ret is False:
                continue

            # 这里进行像素截取 取到屏幕位置 应当是一个固定的位置
            # 下面的所用到的image应当是截取之后的
            # 最后不存储的话 应该返回一个img的数组
            exposure_img = cv2.multiply(image, np.array([alpha]))
            img2gray = cv2.cvtColor(exposure_img, cv2.COLOR_BGR2GRAY)

            cv2.imshow("Capture", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()


class HandelImage:
    def __init__(self, image_name, kernel_n, iter_n=1):
        self.img = cv2.imread(image_name, 0)
        self.iter_n = iter_n
        self.kernel = np.ones((kernel_n, kernel_n), np.uint8)

# 黑底白字情况下 把白色区域缩小 反之放大
    def img_erode(self):
        erosion = cv2.erode(self.img, self.kernel, iterations=self.iter_n)
        while True:
            cv2.imshow('erode', erosion)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyWindow('erode')
                print('腐蚀完毕')
                break

# 白底黑字情况下 把黑色区域缩小 反之放大
    def img_dilate(self):
        dilation = cv2.dilate(self.img, self.kernel, iterations=self.iter_n)
        while True:
            cv2.imshow('dilate', dilation)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyWindow('dilate')
                print('膨胀完毕')
                break

# 黑底白字情况下 把白色区域缩小 反之放大
    def img_open(self):
        opening = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, self.kernel)
        while True:
            cv2.imshow('open', opening)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyWindow('open')
                print('开运算完毕')
                break

# 白底黑字情况下 把黑色区域缩小 反之放大
    def img_close(self):
        closing = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, self.kernel)
        while True:
            cv2.imshow('close',closing)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyWindow('close')
                print('闭运算完毕')
                break

# 轮廓测试
    def img_contours(self):
        im = self.img.copy()
        # imggray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        inverse = (255-im)
        dilation = cv2.dilate(inverse, self.kernel, iterations=self.iter_n)
        # 阈值二值化
        ret, threshold = cv2.threshold(inverse, 127, 255, 0)
        image, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img = cv2.drawContours(self.img, contours, 3, (254, 252, 0), 3)
        while True:
            cv2.imshow('ori', im)
            cv2.imshow('inv', inverse)
            cv2.imshow('dil', dilation)
            cv2.imshow('contours', img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                print('轮廓测试完毕')
                break

    def adaptive_thresh(self):
        blurimg = cv2.medianBlur(self.img, 5)
        ret, th1 = cv2.threshold(blurimg, 100, 255, cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(blurimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2 )
        th3 = cv2.adaptiveThreshold(blurimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        titles = ['original image', 'global thresholding v=127', 'MEAN_C', 'GAUSSIAN_C']
        images = [blurimg, th1, th2, th3]
        for i in range(4):
            plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

