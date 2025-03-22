import os
import numpy as np
import cv2


import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter


class Gaze360(Dataset):
    def __init__(self, path, root, transform, angle, binwidth, train=True):
        self.transform = transform
        self.root = root
        self.orig_list_len = 0
        self.angle = angle
        if train==False:
          angle=90
        self.binwidth=binwidth
        self.lines = []
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    print("here")
                    line = f.readlines()
                    line.pop(0)
                    self.lines.extend(line)
        else:
            with open(path) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len = len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[5]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)
                    
                        
        print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines), angle))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")

        face = line[0]
        lefteye = line[1]
        righteye = line[2]
        name = line[3]
        gaze2d = line[5]
        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        pitch = label[0]* 180 / np.pi
        yaw = label[1]* 180 / np.pi

        img = Image.open(os.path.join(self.root, face))

        # fimg = cv2.imread(os.path.join(self.root, face))
        # fimg = cv2.resize(fimg, (448, 448))/255.0
        # fimg = fimg.transpose(2, 0, 1)
        # img=torch.from_numpy(fimg).type(torch.FloatTensor)

        if self.transform:
            img = self.transform(img)        
        
        # Bin values
        bins = np.array(range(-1*self.angle, self.angle, self.binwidth))
        binned_pose = np.digitize([pitch, yaw], bins) - 1

        labels = binned_pose
        cont_labels = torch.FloatTensor([pitch, yaw])
        

        return img, labels, cont_labels, name

# MPIIGaze의 label파일을 보면 첫 줄이 헤더고, 두번째 줄부터 데이터가 쭉 나열되어 있음.
class Mpiigaze(Dataset): 
  def __init__(self, pathorg, root, transform, train, angle, fold=0):
    self.transform = transform  # 이미지에 적용할 전처리(transformations)
    self.root = root          # 이미지 파일이 있는 폴더 경로
    self.orig_list_len = 0    # 원본 라벨 파일에서 읽은 전체 라인 수
    self.lines = []           # 사용할 데이터가 저장될 리스트
    path=pathorg.copy()       # pathorg를 복사하여 path에 저장
    
    if train==True:           # train이 True일 경우, 현재 fold는 제외
      path.pop(fold)
    else:
      path=path[fold]
      
    if isinstance(path, list):
        for i in path:
            with open(i) as f:
                lines = f.readlines() # 라벨 파일을 라인별로 읽어 리스트로 저장
                lines.pop(0)          # 첫번째 라인은 헤더이므로 제외
                self.orig_list_len += len(lines)  # 전체 라인 수 누적
                for line in lines:
                    gaze2d = line.strip().split(" ")[7] # 8번째 요소(gaze2d)를 읽어옴
                    label = np.array(gaze2d.split(",")).astype("float") # 문자열 -> float 변환
                    # pitch, yaw를 도 단위로 변환하여 각도가 angle 이하인 경우만 데이터셋에 추가
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)
    else:
      with open(path) as f:
        lines = f.readlines()
        lines.pop(0)
        self.orig_list_len += len(lines)
        for line in lines:
            gaze2d = line.strip().split(" ")[7]
            label = np.array(gaze2d.split(",")).astype("float")
            if abs((label[0]*180/np.pi)) <= 42 and abs((label[1]*180/np.pi)) <= 42:
                self.lines.append(line)
                
    # 사용할 데이터셋의 라인 수 출력
    print("Using {} lines from dataset".format(len(self.lines)))
    # 몇 개의 라인이 필터링되었는지 출력
    print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines), angle))
        
  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    name = line[3]
    gaze2d = line[7]
    head2d = line[8]
    lefteye = line[1]
    righteye = line[2]
    face = line[0]

    label = np.array(gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)


    pitch = label[0]* 180 / np.pi
    yaw = label[1]* 180 / np.pi

    img = Image.open(os.path.join(self.root, face))

    # fimg = cv2.imread(os.path.join(self.root, face))
    # fimg = cv2.resize(fimg, (448, 448))/255.0
    # fimg = fimg.transpose(2, 0, 1)
    # img=torch.from_numpy(fimg).type(torch.FloatTensor)
    
    if self.transform:
        img = self.transform(img)        
    
    # Bin values
    bins = np.array(range(-42, 42,3))
    binned_pose = np.digitize([pitch, yaw], bins) - 1

    labels = binned_pose
    cont_labels = torch.FloatTensor([pitch, yaw])


    return img, labels, cont_labels, name


