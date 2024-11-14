import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# 读取图片
image_path = 'D:/1.png'
img = cv2.imread(image_path)

# 提取图像大小
image_size = img.shape[:2]

# 提取颜色特征
average_color = np.mean(img, axis=(0, 1))

# 转换图像为灰度
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 提取形状特征
_, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_area = cv2.contourArea(contours[0])
perimeter = cv2.arcLength(contours[0], True)
circularity = (4 * np.pi * contour_area) / (perimeter ** 2)

# 构建特征向量
features = np.concatenate([np.array(image_size), average_color, [circularity]])

# 假设有多张图片，构建数据集
# 假设有两个类别，0和1
num_samples = 100
data = []
labels = []

for _ in range(num_samples):
    # 生成随机的图片特征
    random_image = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)

    # 提取特征
    image_size = random_image.shape[:2]
    average_color = np.mean(random_image, axis=(0, 1))
    gray_img = cv2.cvtColor(random_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_area = cv2.contourArea(contours[0])
    perimeter = cv2.arcLength(contours[0], True)
    circularity = (4 * np.pi * contour_area) / (perimeter ** 2)

    # 构建特征向量
    features = np.concatenate([np.array(image_size), average_color, [circularity]])

    # 随机分配标签
    label = np.random.randint(0, 2)

    data.append(features)
    labels.append(label)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 初始化K近邻模型
knn_model = KNeighborsClassifier(n_neighbors=3)

# 训练模型
knn_model.fit(X_train, y_train)

# 预测测试集
y_pred = knn_model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")