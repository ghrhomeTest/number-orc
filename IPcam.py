# encoding=utf-8

import cv2
import numpy as np

cap = cv2.VideoCapture("rtmp://192.168.24.13/live/live")
is_open = cap.isOpened()
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# print(is_open)
# print(fps)
# print(size)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter_fourcc('output.avi',fourcc,20.0,(640,480))
while is_open:
    # print('已探测到摄像头')
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # out.write(frame)
    # cv2.imshow('frame',frame)
    cv2.imshow('gray', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()


