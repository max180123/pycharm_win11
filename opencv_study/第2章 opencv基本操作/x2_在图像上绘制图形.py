import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 1 创建空白的图像
img = np.zeros((512, 512, 3), np.uint8)   # 指定二维矩阵512*512

# 2.1绘制直线
# 示例的传参方式为位置传参 关键词传参的关键字并非如下所示
# cv.line(img, start, end, color, thickness)  thickness 线宽
cv.line(img, pt1=(0, 0), pt2=(511, 511), color=(255, 1, 1), thickness=5)

# 2.2绘制圆形
# cv.circle(img, centerpoint, r, color, thickness) thickness = -1为内圆
cv.circle(img, center=(255, 255), radius=200, color=(0, 255, 0), thickness=1)

# 2.3绘制矩形
# cv.rectangle(img, leftupper, rightdown, color, thickness)
cv.rectangle(img, pt1=(15, 15), pt2=(100, 100), color=(0, 0, 255), thickness=5)

# 2.4添加文字
# cv.putText(img, text, station, font, fontsize, color, thickness, cv.LINE_AA)
font = cv.BORDER_CONSTANT
cv.putText(img, "open cv", (10, 100), font, 4, (211, 112, 100), 3, cv.LINE_AA)
# 3 图像展示
plt.imshow(img[:, :, ::-1])
plt.title("plot show"), plt.xticks([]), plt.yticks([])
plt.show()