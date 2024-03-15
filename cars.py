import cv2
import numpy as np

#加载视频
cap = cv2.VideoCapture('C:/Users/YUXINYU/Desktop/1.mp4')

#bgsubmog = cv2.createBackgroundSubtractorMOG()
bgsubmog = cv2.createBackgroundSubtractorMOG2()

# 新建形态学 KERNEL
kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

#逐帧读取视频
while True:
    ret,frame = cap.read()

    if(ret == True):
        # 灰度
        cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # 去噪
        blur = cv2.GaussianBlur(frame,(3,3),5)
        # 去除背景
        mask = bgsubmog.apply(blur)
        # 腐蚀 去掉图中小斑块
        erode = cv2.erode(mask,kernel)
        # 膨胀 还原放大
        dilate = cv2.dilate(erode,kernel,iterations=3)
        # 闭操作 去掉物体内部的小块
        close = cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel)

        # 查找轮廓
        cnts,h = cv2.findContours(close,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for(i, c) in enumerate(cnts):
            (x,y,w,h) = cv2.boundingRect(c)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        #cv2.imshow('video',mask)
        #cv2.imshow('erode',erode)
        cv2.imshow('video', frame)

    key = cv2.waitKey(1)

    # esc 退出视频
    if(key == 27):
        break

# 资源释放
cap.release()
cv2.destroyAllWindows()