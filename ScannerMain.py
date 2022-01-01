import cv2
import numpy as np
import utlis_0

webCamFeed = True
pathImage = "2.jpg"

heightImg = 640
widthImg = 480

# 初始化轨迹栏
utlis_0.initializeTrackbars()
count = 0

while True:
    # 创建空白映像测试
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)

    img = cv2.imread(pathImage)
    # 调整图片大小
    img = cv2.resize(img, (widthImg, heightImg))
    # 把图像转灰度
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    # 获取阈值
    thres = utlis_0.valTrackbars()
    # canny边缘检测
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

    # 检索轮廓
    imgContours = img.copy()
    imgBigContour = img.copy()
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
