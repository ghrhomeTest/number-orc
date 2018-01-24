import cv2
import numpy as np

img = cv2.imread('2510.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
inverse = (255 - gray)
# 高斯BLUR
blur = cv2.GaussianBlur(inverse, (5, 5), 0)
blur2 = cv2.bilateralFilter(inverse, 9, 75, 75)

# 图像二值化 阈值 实测 二值化高斯的好一些
ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(blur2, 127, 255, cv2.THRESH_BINARY)

cv2.imshow('test', inverse)
cv2.imshow('blur', blur)
cv2.imshow('blur2', blur2)
cv2.imshow('thre', thresh)
cv2.imshow('thre2', thresh2)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
