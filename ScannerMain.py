import cv2
import numpy as np
import utlis_0

webCamFeed = True
pathImage = "2.jpg"
heightImg = 640
widthImg = 480

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

