import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F


class L2CS(nn.Module):
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(L2CS, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # nn.Linear는 파이토치에서 사용되는 선형 변환(linear transformation)을 수행하는 클래스로, 
        # Fully Connected Layer 또는 Dense Layer라고도 불립니다.
        # 입력 텐서가 512 * block.expansion개의 차원, 출력 텐서가 num_bins개의 차원
        self.fc_yaw_gaze = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch_gaze = nn.Linear(512 * block.expansion, num_bins)

       # Vestigial layer from previous experiments
       # vestigial layer는 진화 과정에서 사용되지 않게 된 기능이나 구조를 의미합니다.
       # 이전 실험에서 사용되었던 레이어로, 512 * block.expansion개의 차원을 3개의 차원으로 변환합니다.
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules(): 
            # 모든 레이어에 대해 초기화를 수행합니다. 초기화 기법은 Xavier 초기화를 사용합니다.
            # Xavier 초기화란 레이어의 입력과 출력의 분산이 같아지도록 가중치를 초기화하는 방법입니다.
            # 이를 통해 레이어의 출력값이 입력값과 비슷한 분포를 가지도록 만들어 학습을 안정화시킵니다.
            # 아래 코드에선 Conv2d 레이어의 가중치를 Xavier 초기화로 초기화합니다. BatchNorm2d 레이어의 가중치는 1로, 편향은 0으로 초기화합니다.
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n)) # Xavier initialization
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1) # BatchNorm2d의 가중치를 1로 초기화
                m.bias.data.zero_() # BatchNorm2d의 편향을 0으로 초기화

    # 레이어를 만드는 함수입니다.
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # nn.Sequential은 순서대로 모듈을 실행하는 컨테이너입니다.
            # downsample은 입력의 크기를 조정하는 모듈입니다.
            # Conv2d와 BatchNorm2d로 이루어진 downsample을 만듭니다.
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        
        # gaze
        pre_yaw_gaze =  self.fc_yaw_gaze(x)
        pre_pitch_gaze = self.fc_pitch_gaze(x)
        return pre_yaw_gaze, pre_pitch_gaze



