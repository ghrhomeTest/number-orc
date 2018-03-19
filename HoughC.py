import cv2

# 载入并显示图片
img = cv2.imread('tests/mytest/t3.jpg')

# 灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# 输出图像大小，方便根据图像大小调节minRadius和maxRadius
print(img.shape)
img_height = img.shape[0]
print('图片高度'+str(img_height))
# 霍夫变换圆检测
# HoughCircles(image, method, dp, minDist, circles=None, param1=None, param2=None, minRadius=None, maxRadius=None)
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=50, minRadius=2, maxRadius=30)
# cv2.HOUGH_GRADIENT
# cv2.HOUGH_MULTI_SCALE
# cv2.HOUGH_PROBABILISTIC
# cv2.HOUGH_STANDARD
# 输出返回值，方便查看类型
print(circles)
# 输出检测到圆的个数
if circles is not None:
    print(len(circles[0]))

print('-------------我是条分割线-----------------')
# 根据检测到圆的信息，画出每一个圆
if circles is not None:
    for circle in circles[0]:
        if circle[1] > (img_height/3*2):
            # 圆的基本信息
            print(circle[2])
            # 坐标行列
            x = int(circle[0])
            y = int(circle[1])
            # 半径
            r = int(circle[2])
            # 在原图用指定颜色标记出圆的位置
            img = cv2.circle(img, (x, y), r, (0,255,127), -1)
# 显示新图像
cv2.imshow('res', img)

# 按任意键退出
cv2.waitKey(0)
cv2.destroyAllWindows()
