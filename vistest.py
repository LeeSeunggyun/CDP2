import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def draw_gaze(img, pitch, yaw, length=100.0, color=(0, 0, 255)):
    """
    img: 원본 이미지 (BGR)
    pitch, yaw: 라디안 단위
    length: 화살표 길이
    """
    h, w, _ = img.shape
    center = (w // 2, h // 2)

    dx = -length * np.sin(pitch) * np.cos(yaw)
    dy = -length * np.sin(yaw)

    end_point = (int(center[0] + dx), int(center[1] + dy))
    cv2.arrowedLine(img, center, end_point, color, 2, tipLength=0.2)
    return img

def visualize_label(label_path, image_root, index=0):
    """
    label_path: 라벨 텍스트 파일 경로 (e.g., datasets/MPIIFaceGaze/Label/p00.label)
    image_root: 이미지 루트 경로 (e.g., datasets/MPIIFaceGaze/Image/)
    index: 시각화할 라벨 인덱스
    """
    with open(label_path, 'r') as f:
        lines = f.readlines()[1:]  # 첫 줄은 헤더
        line = lines[index].strip().split(' ')
        image_rel_path = line[0]  # face image
        pitch, yaw = map(float, line[7].split(','))  # 라디안 단위

        img_path = os.path.join(image_root, image_rel_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))

        img = draw_gaze(img, pitch, yaw)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(img_rgb)
        plt.title(f"Pitch: {pitch:.2f} rad, Yaw: {yaw:.2f} rad")
        plt.axis('off')
        plt.show()

visualize_label(
    label_path='datasets/MPIIFaceGaze/Label/p14.label',
    image_root='datasets/MPIIFaceGaze/Image/',
    index=1  # 보고 싶은 이미지 인덱스
)
