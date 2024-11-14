import cv2 as cv
import matplotlib.pyplot as plt

# 1 读取图片 img = cv2.imread("路径", 0)  -1彩色 0灰色 1 alpha
# 读取图像，确保使用BGR颜色空间  img = cv.imread("D:/1.png", cv.IMREAD_COLOR) 默认为bgr
img = cv.imread("D:/1.png")

# # 2.1 显示图像 cv2.imshow("图像窗口名称，字符串", 对象)
# cv.imshow("image111", img)
# # 图像停留时间 cv2.waitKey(毫秒)  0 永久
# cv.waitKey(10000)

# 2.2 matplotlib显示        opencv 存储图像（BGR） 所以显示彩色图像需要翻转
# img = cv.imread("D:/1.png")
plt.imshow(img[:, :, ::-1])
plt.show()

# 2.2.1 灰色图片打开的方式
# img = cv.imread("D:/1.png", 0)
# plt.imshow(img, cmap=plt.cm.gray)
# plt.show()

# 3 保存图像 cv2.imwrite（"位置", img）
cv.imwrite('D:/2.png', img)