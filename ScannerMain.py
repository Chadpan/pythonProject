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
    contours, hierarchy = cv2.findContours(
        imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

    # 检索最大轮廓
    biggest, maxArea = utlis_0.biggestContour(contours)
    if biggest.size != 0:
        biggest = utlis_0.reorder(biggest)
        # 绘制轮廓
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
        imgBigContour = utlis_0.drawRectangle(imgBigContour, biggest, 2)
        pts1 = np.float32(biggest)  # 描点
        pts2 = np.float32(
            [[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(
            img, matrix, (widthImg, heightImg))

        # 每侧移除20个像素
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] -
                                        20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

        # 自适应阈值
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        # 显示
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])

    else:
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    # 对应标签
    lables = (['Original', 'Gray', 'Threshold', 'Contours'],
              ['Biggest Contour', 'Warp Prespective', 'Warp Gray', 'Adaptive Threshold'])

    stackedImage = utlis_0.stackImages(imageArray, 0.75, lables)
    cv2.imshow("Result", stackedImage)
