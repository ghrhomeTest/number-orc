import cv2
import numpy as np
import os
from ImageProcessing.OpenCVUtils import inverse_colors, sort_contours

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
CROP_DIR = 'crops'

# 图片处理 主要内容 这里的version='_2_0'


class FrameProcessor:
    def __init__(self, height, version, deci, beishu, debug=False, write_digits=False):
        self.debug = debug
        self.version = version
        self.height = height
        self.deci = deci
        self.beishu = beishu
        self.file_name = None
        self.img = None
        self.width = 0
        self.original = None
        self.write_digits = write_digits

        self.knn = self.train_knn(self.version)

    def set_image(self, file_name):
        self.file_name = file_name
        self.img = cv2.imread(file_name)
        self.original, self.width = self.resize_to_height(self.height)
        self.img = self.original.copy()

    def resize_to_height(self, height):
        r = self.img.shape[0] / float(height)
        dim = (int(self.img.shape[1] / r), height)
        img = cv2.resize(self.img, dim, interpolation=cv2.INTER_AREA)
        return img, dim[0]

    # knn训练
    def train_knn(self, version):
        npa_classifications = np.loadtxt("knn/classifications" + version + ".txt",
                                         np.float32)  # read in training classifications
        npa_flattened_images = np.loadtxt("knn/flattened_images" + version + ".txt",
                                          np.float32)  # read in training images

        npa_classifications = npa_classifications.reshape((npa_classifications.size, 1))
        k_nearest = cv2.ml.KNearest_create()
        k_nearest.train(npa_flattened_images, cv2.ml.ROW_SAMPLE, npa_classifications)
        return k_nearest
    # 图像处理 return debug_images, output

    def process_image(self, blur, threshold, adjustment, erode, iterations):

        self.img = self.original.copy()
        # image 输出
        img_height = self.img.shape[0]
        print(self.img.shape)
        debug_images = []
        # float(2.5)=1.250000 float保留六位小数
        alpha = float(2.5)

        debug_images.append(('Original', self.original))

        # 调整曝光度
        exposure_img = cv2.multiply(self.img, np.array([alpha]))
        debug_images.append(('Exposure Adjust', exposure_img))

        # 灰度转换 这里出来的图像最干净
        img2gray = cv2.cvtColor(exposure_img, cv2.COLOR_BGR2GRAY)
        debug_images.append(('Grayscale', img2gray))

        # 霍夫圆
        circles_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(circles_gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=1,
                                   maxRadius=10)
        if circles is None:
            print('没有圆')
        # print(circles)
        # print(len(circles[0]))
        # 高斯模糊
        img_blurred = cv2.GaussianBlur(img2gray, (blur, blur), 0)
        debug_images.append(('Blurred', img_blurred))

        cropped = img_blurred
        # cropped = img2gray

        # 二值化
        cropped_threshold = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                  threshold, adjustment)
        debug_images.append(('Cropped Threshold', cropped_threshold))

        # 这里 ！！
        # 图像侵蚀 问题应该在这里 把黑色区域放大 与小数点连起来了
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode, erode))
        eroded = cv2.erode(cropped_threshold, kernel, iterations=iterations)
        debug_images.append(('Eroded', eroded))

        # 图像反转
        inverse = inverse_colors(eroded)
        debug_images.append(('Inversed', inverse))

        # 描画轮廓 findContours返回三个参数 image,contours,hierarchy
        # cv2.RETR_EXTERNAL 检测最外围轮廓
        # cv2.CHAIN_APPROX_NONE 保存物体边界上所有连续的轮廓点到contours向量里
        # 这个函数只能从黑底白字的图里找 即 寻找的东西是白色的 而背景是黑色的 必须是二值化之后的图
        _, contours, _ = cv2.findContours(inverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # get contours 这里应该已经得到轮廓向量

        # Assuming we find some, we'll sort them in order left -> right 从左到右排序
        if len(contours) > 0:
            contours, _ = sort_contours(contours)
        # 小数点
        potential_decimals = []
        # 数字
        potential_digits = []

        total_digit_height = 0
        total_digit_y = 0

        # 希望得到的形状 0.6
        desired_aspect = 0.55
        # 1的高宽比
        digit_one_aspect = 0.3
        # 缓冲区 原来是0.15
        aspect_buffer = 0.15

        # 矩形框出来所有数字和小数点
        for contour in contours:
            # 得到一个斜的框 x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
            [x, y, w, h] = cv2.boundingRect(contour)

            aspect = float(w) / h
            size = w * h

            # 方形的就作为一个小数点
            # 这里应该要修改
            if size > 100 and 1 - .3 <= aspect <= 1 + .3:
                potential_decimals.append(contour)

            #  如果很小且不是方的 重新跑
            if size < 20 * 100 and 1 + aspect_buffer < aspect < 1 - aspect_buffer:
                continue

            # 忽略任何宽度大大与高度的长方形
            if w > h:
                if self.debug:
                    # 这里是描绘出最小矩形
                    cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                continue

            #  如果轮廓的尺寸合适则保存
            if ((size > 2000) and (desired_aspect - aspect_buffer <= aspect <= desired_aspect + aspect_buffer) or
                (size > 1000) and (digit_one_aspect - aspect_buffer <= aspect <= digit_one_aspect + aspect_buffer)):
                # Keep track of the height and y position so we can run averages later
                total_digit_height += h
                total_digit_y += y
                potential_digits.append(contour)
            else:
                if self.debug:
                    cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        avg_digit_height = 0
        avg_digit_y = 0
        potential_digits_count = len(potential_digits)
        left_most_digit = 0
        right_most_digit = 0
        digit_x_positions = []

        # 计算高度和Y坐标的平均值
        if potential_digits_count > 0:
            avg_digit_height = float(total_digit_height) / potential_digits_count
            avg_digit_y = float(total_digit_y) / potential_digits_count
            if self.debug:
                print("Average Digit Height and Y: " + str(avg_digit_height) + " and " + str(avg_digit_y))
        # 定义输出的字符串
        output = ''
        ix = 0
        # 算法主要的调整部分
        # Loop over all the potential digits and see if they are candidates to run through KNN to get the digit
        # 这里是把数字分别识别出来存入数组
        for pot_digit in potential_digits:
            # 画出矩形
            [x, y, w, h] = cv2.boundingRect(pot_digit)

            # Does this contour match the averages
            if avg_digit_height * 0.2 <= h <= avg_digit_height * 1.2 and \
                    avg_digit_y * 0.2 <= y <= avg_digit_height * 1.2:
                # Crop the contour off the eroded image
                # 这里是从eroded裁剪的 试试看换个裁剪方式
                cropped = eroded[y:y + h, x: x + w]
                # Draw a rect around it 画个矩形 对象，左上角，右下角 颜色 通道数
                cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                debug_images.append(('digit' + str(ix), cropped))

                # Call KNN to determine the digit
                digit = self.predict_digit(cropped)
                if self.debug:
                    print("Digit=======: " + digit)
                output += digit

                # Helper code to write out the digit image file for use in KNN training  非训练模式下不运行
                if self.write_digits:
                    # 默认为false 此为写入图片
                    _, full_file = os.path.split(self.file_name)
                    file_name = full_file.split('.')
                    crop_file_path = CROP_DIR + '/' + digit + '_' + file_name[0] + '_crop_' + str(ix) + '.png'
                    cv2.imwrite(crop_file_path, cropped)

                # Track the x positions of where the digits are  追踪X的位置
                # 每个数字分别作为一个图 每个图的左上角的坐标其中的横坐标为X
                if left_most_digit == 0 or x < left_most_digit:
                    left_most_digit = x
                # 每个框的最右横坐标
                if right_most_digit == 0 or x > right_most_digit:
                    right_most_digit = x + w

                digit_x_positions.append(x)

                ix += 1
            else:
                if self.debug:
                    cv2.rectangle(self.img, (x, y), (x + w, y + h), (66, 146, 244), 2)

        decimal_x = 0
        # Loop over the potential digits and find a square that's between the left/right digit x positions on the
        # lower half of the screen
        # 这里是识别小数点 并把位置算好存到数字中间
        # x,y 是外框左上角坐标

        # 这里用霍夫圆替代 绘制出圆 计算出圆的圆心与半径 由圆心坐标得出此小数点的位置
        # 具体为 使用某一个处理过的图像 检测其靠下部分的圆形位置 返回此圆的圆心坐标与半径
        # for pot_decimal in potential_decimals:
        #     # 把所有潜在的小数点框出来 是方的
        #     [x, y, w, h] = cv2.boundingRect(pot_decimal)
        #     # 横坐标是在左右之间 并 是在下半部分
        #     # y坐标从上到下递增
        #     if left_most_digit < x < right_most_digit and y > (self.height / 2):
        #         # 画出外框 可省略
        #         cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #         # 这里是关键 小数点的横坐标
        #         decimal_x = x

        # 以下为寻找圆形小数点 成了 但是还有其他的前期工作 例如图像的裁剪
        # for circle in circles[0]:
        #     if circle[1] > (img_height/3*2):
        #         print(circle[2])
        #         # 坐标行列
        #         cir_x = int(circle[0])
        #         cir_y = int(circle[1])
        #         # 半径
        #         cir_r = int(circle[2])
        #         if left_most_digit < cir_x < right_most_digit and cir_y > (self.height / 2):
        #             decimal_x = cir_x
        # 这里寻找完毕 得到了小数点圆心的横坐标 用于下面的对比
        # 下面这行是测试用的 并且测试成功
        decimal_x = self.deci/self.beishu

        # Once we know the position of the decimal, we'll insert it into our string
        for ix, digit_x in enumerate(digit_x_positions):
            if digit_x > decimal_x:
                # insert
                output = output[:ix] + '.' + output[ix:]
                break

        # Debugging to show the left/right digit x positions
        if self.debug:
            cv2.rectangle(self.img, (left_most_digit, int(avg_digit_y)),
                          (left_most_digit + right_most_digit - left_most_digit,
                           int(avg_digit_y) + int(avg_digit_height)),
                          (66, 244, 212), 2)

        # Log some information
        if self.debug:
            print("Potential Digits " + str(len(potential_digits)))
            print("Potential Decimals " + str(len(potential_decimals)))
            print("String: " + output)

        return debug_images, output

    # Predict the digit from an image using KNN
    def predict_digit(self, digit_mat):
        # Resize the image
        imgROIResized = cv2.resize(digit_mat, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        # Reshape the image
        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        # Convert it to floats
        npaROIResized = np.float32(npaROIResized)
        _, results, neigh_resp, dists = self.knn.findNearest(npaROIResized, k=1)
        predicted_digit = str(chr(int(results[0][0])))
        if predicted_digit == 'A':
            predicted_digit = '.'
        return predicted_digit
