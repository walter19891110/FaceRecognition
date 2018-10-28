import face_recognition
import cv2
import time
import dlib
import datetime
import numpy as np
import pandas as pd


def my_compare_faces(face_encodings, face_to_compare, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param face_encodings: A list of known face encodings
    :param face_to_compare: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    result = list()
    if len(face_encodings) == 0:
        return result
    face_dis = np.linalg.norm(np.array(face_encodings) - np.array(face_to_compare), axis=1)
    b = np.argsort(face_dis)
    min_index = b[0]  # 最小元素的索引
    if face_dis[min_index] <= tolerance:  # 最小值符合阈值，说明找到了人, 返回索引
        result.append(min_index)

    return result


def face_detect():
    # 获取摄像头，其参数0表示第一个摄像头，一般就是笔记本的内建摄像头
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 500)  # 设置摄像窗口的宽
    video_capture.set(4, 500)  # 设置摄像窗口的高

    # 加载一个图片，并进行训练，提取特征值
    obama_image = face_recognition.load_image_file("./image/trainfaces/wangjing/wangjing.jpg")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    # 加载另一个图片，并进行训练，提取特征值
    biden_image = face_recognition.load_image_file("./image/trainfaces/wangjing/wangjizhuo.jpg")
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
    # 创建人脸特征库，以及对应的名字
    known_face_encodings = [
        biden_face_encoding,
        obama_face_encoding
    ]
    known_face_names = [
        'wangjizhuo',
        'wangjing'
    ]

    face_detector = dlib.get_frontal_face_detector()  # 设置人脸检测器,使用hog算法进行人脸检测
    n = 0
    total = 0
    while True:
        start = time.time()
        # 读取摄像头画面
        ret, frame = video_capture.read()

        # 改变摄像头图像的大小，改为1/2,图像小，所做的计算就少,加快识别速度
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # opencv的图像是BGR格式的，而人脸识别需要是的RGB格式的，因此需要进行一个转换
        rgb_frame = small_frame[:, :, ::-1]
        # 识别视频中的人脸，并提取特征值
        start = time.time()
        # face_locations = face_recognition.face_locations(rgb_frame, model="cnn")  # face_recognition的库函数太慢了，暂不使用

        face_locations = [(face.top(), face.right(), face.bottom(), face.left()) for face in face_detector(rgb_frame)]
        print("人脸检测时间：", time.time() - start)
        start = time.time()
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)  # 使用resNet提取特征值，共128个特征点
        end = time.time() - start
        n += 1
        total += end
        if n >= 100:
            print("特征提取时间：", total/n)
            n = 0
            total = 0

        # 对视频中的每张人脸进行识别
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # 显示的时候，还原回来
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            # 对比人脸库，使用距离公式，距离小于阈值，即认为是同一人，如果找到，就返回true,没有就返回false
            """
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = r'Unknown'
            ma = face_recognition.face_distance(known_face_encodings, face_encoding)
            print(type(known_face_names), type(ma))
            # 如果识别到人脸，返回识别的第一个人名
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            """
            match = my_compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if len(match) != 0:
                name = known_face_names[match[0]]
            # print(name)
            # 在视频中画个人脸框
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # 在人脸框下，画个标签写上人名
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # 显示画面
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # print("总时间：", time.time() - start)

    video_capture.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 关闭所有图像窗口


if __name__ == '__main__':
    face_detect()
