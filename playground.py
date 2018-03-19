import cv2
import time
import sys
from ImageProcessing import FrameProcessor, ProcessingVariables
from DisplayUtils.TileDisplay import show_img, reset_tiles
from UsbVideo import circle_position


window_name = 'Playground'
# 传入文件要求白底黑字 2510.png ppp.jpg 暂时是这两个测试文件
# file_name = 'tests/single_line/2510.png'
# 如果传入的是原始图像 则进入

# file_name = 'C:/Users/Administrator/Pictures/2.png'
# file_name = 'samples/shell_berlin/test11.jpg'
version = '_2_0'

# 设置面板的滑动条数值
erode = ProcessingVariables.erode
threshold = ProcessingVariables.threshold
adjustment = ProcessingVariables.adjustment
iterations = ProcessingVariables.iterations
blur = ProcessingVariables.blur
std_height = 90
debug = False


# 主程序
def main(file_name):
    # 载入图像
    deci1, beishu1 = circle_position(file_name)
    frameProcessor = FrameProcessor(std_height, version, deci1, beishu1, debug)
    img_file = file_name
    if len(sys.argv) == 2:
        img_file = sys.argv[1]
    # 设置playground的UI
    if debug:
        setup_ui()
    # 设置图像参数
    frameProcessor.set_image(img_file)
    # result_str = ''
    # print(result_str)
    # 处理图像
    # process_image()
    reset_tiles()
    start_time = time.time()
    # 处理图像
    debug_images, output = frameProcessor.process_image(blur, threshold, adjustment, erode, iterations)
    result_str = output
    #result_str = process_image()
    # print(result_str)
    # 等待 这里应该在
    # CaptureVideo.cap_video()
    # cv2.waitKey()
    return result_str


def process_image():
    # 重置栈
    reset_tiles()
    start_time = time.time()
    # 处理图像
    debug_images, output = frameProcessor.process_image(blur, threshold, adjustment, erode, iterations)

    # 显示处理过程
    # for image in debug_images:
    #     show_img(image[0], image[1])
    # if debug:
    #     print("Processed image in %s seconds" % (time.time() - start_time))
    #     cv2.imshow(window_name, frameProcessor.img)
    #     cv2.moveWindow(window_name, 600, 600)
    return output


def setup_ui():
    # 创建窗口 命名为 window_name
    cv2.namedWindow(window_name)
    # 创建滑动条 初始值在ProcessingVariables.py里
    # 第一个参数是滑动条的名字 第二个是滑动条所在的窗口名字，第三个是默认位置
    # 第四个是最大值，第五个是回调函数
    cv2.createTrackbar('Threshold', window_name, int(threshold), 500, change_threshold)
    cv2.createTrackbar('Iterations', window_name, int(iterations), 5, change_iterations)
    cv2.createTrackbar('Adjust', window_name, int(adjustment), 200, change_adj)
    # 先把erode去掉
    cv2.createTrackbar('Erode', window_name, int(erode), 5, change_erode)
    cv2.createTrackbar('Blur', window_name, int(blur), 25, change_blur)


def change_blur(x):
    # 滤波 模糊图像
    global blur
    print('blur: ' + str(x))
    if x % 2 == 0:
        x += 1
    blur = x
    process_image()


def change_adj(x):
    # 亮度调节
    global adjustment
    print('Adjust: ' + str(x))
    adjustment = x
    process_image()


def change_erode(x):
    # 图像膨胀
    global erode
    print('Erode: ' + str(x))
    erode = x
    process_image()


def change_iterations(x):
    # 迭代次数
    print('Iterations: ' + str(x))
    global iterations
    iterations = x
    process_image()


def change_threshold(x):
    # 图像阈值 二值化
    print('Threshold: ' + str(x))
    global threshold

    if x % 2 == 0:
        x += 1
    threshold = x
    process_image()


if __name__ == "__main__":
    main()
