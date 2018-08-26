---
layout: post
title: SaveCameraVideo
date: 2018-08-26 15:00 +0800
categories: image
tags: opencv
---  
{% highlight python linenos %}

#encoding:utf-8
import numpy as np
import cv2

cap = cv2.VideoCapture(0)#参数为0，打开内置摄像头,从文件中播放视频，填入文件名,播放视频延时一般设为25ms
fourcc = cv2.VideoWriter_fourcc(*'XVID')#指定FourCC编码，一般是ID，MJPG是高尺寸视频，X264得到小尺寸视频

out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))#播放频率和帧的大小，最后是isColor标签True为彩色
#cap.get(num)取视频的一些参数信息num为0-18的任何数，每一个数代表一个属性
while(cap.isOpened()):#检查是否成功初始化
    ret, frame = cap.read()#返回两个值,ret为True,False，代表是否读取到图片,frame表示截取到一帧的图片
    if ret == True:
        frame = cv2.flip(frame, 0)

        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):#64位系统需要加上0xFF，按q退出
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
{% endhighlight %}