# 图是物体反射光、透射光的分布
# 像是人的视觉系统所接受的图在人脑中所形成的印象、认识
# 模拟图像：连续存储的数据（易受干扰） 灰度图片
# 数字图像：分级存储的数据 彩色图片
# 图像分类：  位数：图像的表示，常见的是八位  2**8=256  共256级 0最黑 1最白
#           分类：二值图像、灰度图像、彩色图像
# 测试
import cv2
lena = cv2.imread("D:/1.png")
cv2.imshow("image", lena)
cv2.waitKey(0)