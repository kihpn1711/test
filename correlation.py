import numpy as np
import cv2
import matplotlib.pyplot as plt

from collections import Counter

def horizontalFig(image_path, hang, cot):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Không thể đọc được ảnh từ đường dẫn đã cung cấp.")

    height, width = image.shape
    ox = []
    oy = []
    num = 0

    # Bước 2: Random 30 giá trị hàng khác nhau
    xx = np.random.choice(height - 2, hang, replace=False)
    yy = np.random.choice(width - 1, cot, replace=False)
    for i in range(len(xx)):
        for j in range(len(yy)):
            if xx[i] + 1 >= height:
                continue
            ox.append(image[xx[i]][yy[j]])
            oy.append(image[xx[i] + 1][yy[j]])


    plt.scatter(ox, oy, color='blue')
    plt.xlabel('pixel value on (x,y)')
    plt.ylabel('pixel value on (x + 1, y)')
    plt.show()

def VerticalFig(image_path, hang, cot):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Không thể đọc được ảnh từ đường dẫn đã cung cấp.")

    height, width = image.shape
    ox = []
    oy = []
    num = 0

    # Bước 2: Random 30 giá trị hàng khác nhau
    xx = np.random.choice(height - 1, hang, replace=False)
    yy = np.random.choice(width - 2, cot, replace=False)
    for i in range(len(xx)):
        for j in range(len(yy)):
            if yy[j] + 1 >= height:
                continue
            ox.append(image[xx[i]][yy[j]])
            oy.append(image[xx[i]][yy[j] + 1])


    plt.scatter(ox, oy, color='blue')
    plt.xlabel('pixel value on (x,y)')
    plt.ylabel('pixel value on (x + 1, y)')
    plt.show()

def horizontalFig(image_path, hang, cot):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Không thể đọc được ảnh từ đường dẫn đã cung cấp.")

    height, width = image.shape
    ox = []
    oy = []
    num = 0

    # Bước 2: Random 30 giá trị hàng khác nhau
    xx = np.random.choice(height - 2, hang, replace=False)
    yy = np.random.choice(width - 2, cot, replace=False)
    for i in range(len(xx)):
        if xx[i] + 1 >= height:
            continue
        for j in range(len(yy)):
            if yy[j] + 1 >= height:
                continue
            ox.append(image[xx[i]][yy[j]])
            oy.append(image[xx[i] + 1][yy[j] + 1])


    plt.scatter(ox, oy, color='blue')
    plt.xlabel('pixel value on (x,y)')
    plt.ylabel('pixel value on (x + 1, y)')
    plt.show()

def entropy(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
    else:
        entropy = 0.0
        p = {}
        height, width, channels = image.shape

        # Duyệt qua từng pixel của ảnh
        for y in range(height):
            for x in range(width):
                # Lấy giá trị màu tại vị trí (x, y)
                for z in range(channels):
                    pixel_value = image[y, x, z]
                    if pixel_value not in p:
                        p[pixel_value] = 1
                    else:
                        p[pixel_value] += 1
        result_dict = {key: value / (height * width * channels) for key, value in p.items()}
        for _, value in result_dict.items():
            entropy += value * np.log2(1 / value)
        return entropy


