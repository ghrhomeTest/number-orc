import cv2
import numpy as np
from result import thismain
import threading
import time


# file_name = 'tests/mytest/t7.jpg'
path = 'tests/mytest/'
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
# 这里进行ret的判断
# 这段是视频的测试 暂时不用


def show_p(image, win_name='default'):
    cv2.imshow(win_name, image)


def fun_timer():
    nowtime = time.strftime('%d-%H-%M-%S', time.localtime())
    print(nowtime)
    mypic(nowtime)
    global timer
    timer = threading.Timer(time_interval, fun_timer)
    timer.start()


time_interval = 1800
timer = threading.Timer(time_interval, fun_timer)
timer.start()
# 记得写到循环定时器里


def mypic(name):
    # while cap.isOpened():
    ret, img = cap.read()

    # if ret is False:
    #     continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    reverse = 255-gray
    # _, thth = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    _, thth = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)

    rev = 255 - thth
    # 735 315
    caij = rev[70:550, 300:1200]
    ccc = img[70:550, 300:1200]
    med1 = cv2.medianBlur(caij, 5)
    # show_p(ccc, '1')
    # # show_p(caij, '4')
    # # show_p(med1, '3')
    # # show_p(gray, '2')
    # # show_p(reverse, '1')
    #
    # cv2.waitKey(0)
    cv2.imwrite(path+name+'.jpg', ccc)
    thismain(path+name+'.jpg', name)
    # if cv2.waitKey(0) & 0xFF == ord('q'):q
    #     # 图片名使用时间命名 格式为 日-时-分-秒
    #     # cv2.imwrite(path+'t8.jpg', ccc)
    #     thismain(path+'t8.jpg')
    #     timer.cancel()
    #     # break


# if 0xFF == ord('q'):
#     timer.cancel()
#     print('结束了')


# r, im = cap.read()
#
# debug_imgs=[]
# kenerl = np.ones((9, 9), np.uint8)
# debug_imgs.append((im, 'orgin'))
#
# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# debug_imgs.append((gray, 'gray'))
#
# blur = cv2.GaussianBlur(gray, (5, 5), 0)
# debug_imgs.append((blur, 'blur'))
#
# reverse = 255-blur
# debug_imgs.append((reverse, 'reverse'))
#
# _, th = cv2.threshold(gray, 127, 255, 0)
# debug_imgs.append((th, 'threshold'))
#
# _, rth = cv2.threshold(reverse, 127, 255, 0)
# debug_imgs.append((rth, 'rth'))
#
# dilation = cv2.dilate(th, kenerl, iterations=1)
# debug_imgs.append((dilation, 'dilation'))
#
# _, con, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(im, con, -1, (0, 255, 0), 1)
#
# # 展示图片
# # for i in debug_imgs:
# #     show_p(i[0], i[1])
# show_p(im)
#
# if cv2.waitKey(0) & 0xFF == ord('q'):
#     print('DONE!')
#     #cv2.imwrite(path+'or.jpg', rth)
#     cap.release()
#     cv2.destroyAllWindows()
