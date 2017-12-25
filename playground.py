import cv2
import time
import sys
#测试提交
from ImageProcessing import FrameProcessor, ProcessingVariables
from DisplayUtils.TileDisplay import show_img, reset_tiles

window_name = 'Playground'
file_name = 'tests/single_line/46A29Z.jpg'
#file_name = 'C:/Users/Administrator/Pictures/2.png'
#file_name = 'samples/shell_berlin/test11.jpg'
version = '_2_0'

#设置面板的滑动条数值
erode = ProcessingVariables.erode
threshold = ProcessingVariables.threshold
adjustment = ProcessingVariables.adjustment
iterations = ProcessingVariables.iterations
blur = ProcessingVariables.blur

std_height = 90

frameProcessor = FrameProcessor(std_height, version, True)

#主程序
def main():
    #载入图像
    img_file = file_name
    if len(sys.argv) == 2:
        img_file = sys.argv[1]
    #设置playground的UI
    setup_ui()
    #设置图像参数
    frameProcessor.set_image(img_file)
    #处理图像
    process_image()
    #等待
    cv2.waitKey()


def process_image():
    #重置栈
    reset_tiles()
    start_time = time.time()
    #处理图像
    debug_images, output = frameProcessor.process_image(blur, threshold, adjustment, erode, iterations)

    for image in debug_images:
        show_img(image[0], image[1])

    print("Processed image in %s seconds" % (time.time() - start_time))

    cv2.imshow(window_name, frameProcessor.img)
    cv2.moveWindow(window_name, 600, 600)


def setup_ui():
    #创建窗口 命名为 window_name
    cv2.namedWindow(window_name)
    # 创建滑动条 初始值在ProcessingVariables.py里
    # 第一个参数是滑动条的名字 第二个是滑动条所在的窗口名字，第三个是默认位置
    # 第四个是最大值，第五个是回调函数
    cv2.createTrackbar('Threshold', window_name, int(threshold), 500, change_threshold)
    cv2.createTrackbar('Iterations', window_name, int(iterations), 5, change_iterations)
    cv2.createTrackbar('Adjust', window_name, int(adjustment), 200, change_adj)
    cv2.createTrackbar('Erode', window_name, int(erode), 5, change_erode)
    cv2.createTrackbar('Blur', window_name, int(blur), 25, change_blur)


def change_blur(x):
    global blur
    print('Adjust: ' + str(x))
    if x % 2 == 0:
        x += 1
    blur = x
    process_image()


def change_adj(x):
    global adjustment
    print('Adjust: ' + str(x))
    adjustment = x
    process_image()


def change_erode(x):
    global erode
    print('Erode: ' + str(x))
    erode = x
    process_image()


def change_iterations(x):
    print('Iterations: ' + str(x))
    global iterations
    iterations = x
    process_image()


def change_threshold(x):
    print('Threshold: ' + str(x))
    global threshold

    if x % 2 == 0:
        x += 1
    threshold = x
    process_image()


if __name__ == "__main__":
    main()
