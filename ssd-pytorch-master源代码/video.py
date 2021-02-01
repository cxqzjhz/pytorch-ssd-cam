#-------------------------------------#
#           调用摄像头检测
#-------------------------------------#
from ssd import SSD
from PIL import Image
import numpy as np
import cv2
import time
ssd = SSD()
# 调用摄像头
capture=cv2.VideoCapture(0) # capture=cv2.VideoCapture("1.mp4")
# capture=cv2.VideoCapture("毕业晚会.MP4")
fps = 0.0
while(True):
    t1 = time.time()  # 获取当前时间戳,秒为单位
    # 读取某一帧
    ref, frame=capture.read()
    if not ref:
        print('视频文件读取结束...\n按下任意键结束程序...')
        cv2.waitKey(0)
        break
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))

    # 进行检测
    frame = np.array(ssd.detect_image(frame))

    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps={0:.2f}".format(fps))
    frame = cv2.putText(frame, "fps={0:.2f}".format(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("video",frame)
    c = cv2.waitKey(1) & 0xff 
    if c == 27: # 按下Esc键退出
        print('用户按下Esc键,程序提前退出...')
        break
capture.release()
cv2.destroyAllWindows()
