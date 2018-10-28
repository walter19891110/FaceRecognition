#!/usr/bin/python
# -*- coding=utf-8 -*-

"""face detect
https://github.com/seathiefwang/FaceRecognition-tensorflow
http://tumumu.cn/2017/05/02/deep-learning-face/
"""
# pylint: disable=invalid-name
import os
import random
import numpy as np
import cv2
import dlib
import time


def createdir(*args):
    ''' create dir'''
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)

IMGSIZE = 64


def getpaddingSize(shape):
    ''' get size to make image to be a square rect '''
    h, w = shape
    longest = max(h, w)
    result = (np.array([longest]*4, int) - np.array([h, h, w, w], int)) // 2
    return result.tolist()

def dealwithimage(img, h=64, w=64):
    ''' dealwithimage '''
    #img = cv2.imread(imgpath)
    top, bottom, left, right = getpaddingSize(img.shape[0:2])
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.resize(img, (h, w))
    return img


def relight(imgsrc, alpha=1, bias=0):
    '''relight'''
    imgsrc = imgsrc.astype(float)
    imgsrc = imgsrc * alpha + bias
    imgsrc[imgsrc < 0] = 0
    imgsrc[imgsrc > 255] = 255
    imgsrc = imgsrc.astype(np.uint8)
    return imgsrc


def getfacefromcamera(outdir):
    """人脸数据采集，获取人脸数据，用于模型训练"""
    cv2_dir = cv2.__file__  # cv2库文件的目录
    haarcascade_data = os.path.abspath(os.path.join(cv2_dir, "../data"))
    createdir(outdir)
    camera = cv2.VideoCapture(0)  # 其参数0表示第一个摄像头，一般就是笔记本的内建摄像头
    haar = cv2.CascadeClassifier(haarcascade_data + '/haarcascade_frontalface_default.xml')
    n = 1
    while 1:
        if n <= 50:
            print('It`s processing %s image.' % n)
            # 读帧
            success, img = camera.read()

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = haar.detectMultiScale(gray_img, 1.3, 5)
            for f_x, f_y, f_w, f_h in faces:
                face = img[f_y:f_y+f_h, f_x:f_x+f_w]
                face = cv2.resize(face, (IMGSIZE, IMGSIZE))
                #could deal with face to train
                face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                cv2.imwrite(os.path.join(outdir, str(n)+'.jpg'), face)

                cv2.putText(img, 'haha', (f_x, f_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  #显示名字
                img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
                n+=1
            cv2.imshow('img', img)
            key = cv2.waitKey(30) & 0xff  # 按q键退出
            if key == 27:
                break
        else:
            break
    camera.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 关闭所有图像窗口

def cam_test():
    """测试摄像头"""
    cv2_dir = cv2.__file__  # cv2库文件的目录
    haarcascade_data = os.path.abspath(os.path.join(cv2_dir, "../data"))
    camera = cv2.VideoCapture(0)  # 其参数0表示第一个摄像头，一般就是笔记本的内建摄像头
    haar = cv2.CascadeClassifier(haarcascade_data + '/haarcascade_frontalface_default.xml')
    while True:
        # 读帧
        success, img = camera.read()

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray_img, 1.3, 5)
        for f_x, f_y, f_w, f_h in faces:
            face = img[f_y:f_y + f_h, f_x:f_x + f_w]
            face = cv2.resize(face, (IMGSIZE, IMGSIZE))
            # could deal with face to train
            face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))

            cv2.putText(img, 'haha', (f_x, f_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  # 显示名字
            img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)

        cv2.imshow('img', img)
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 关闭所有图像窗口


def dlib_test():
    detector = dlib.get_frontal_face_detector()
    cam = cv2.VideoCapture(0)
    color_green = (0, 255, 0)
    line_width = 3
    while True:
        ret_val, img = cam.read()
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        start = time.time()
        dets = detector(rgb_image)
        print(type(dets), dets)
        print("人脸检测时间：", time.time() - start)
        for det in dets:
            cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # name = input('please input yourename: ')
    print(cv2.__file__)
    print(os.path.abspath(os.path.join(cv2.__file__, "../data")))
    # getfacefromcamera(os.path.join('./image/trainfaces', 'wangjing'))
    dlib_test()
