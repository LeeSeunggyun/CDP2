import os
import argparse
import time

import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from l2cs import L2CS, select_device, Gaze360, Mpiigaze

#  명령어 예시
#  (복붙용)python train.py --dataset mpiigaze --output output/snapshots/ --gpu 0 --num_epochs 50 --batch_size 16 --arch ResNet50 --alpha 1 --lr 0.00001
#  python train.py \
#  --dataset mpiigaze \ # 데이터셋 이름
#  --output output/snapshots/ \ # 모델 저장 경로
#  --snapshot '' \ # 학습된 모델(스냅샷) 경로
#  --gpu 0 \ # 사용할 GPU 아이디
#  --num_epochs 50 \ # 논문과 같은 epoch(아래 코드에선 default=60인데, 논문에서는 50)
#  --batch_size 16 \ # 논문과 같은 batch_size
#  --arch ResNet50 \ # 논문과 같은 모델
#  --alpha 1 \
#  --lr 0.00001 \ # 논문과 같은 learning_rate
# 이게 훈련이 leave-one-person-out으로 되는데, 이게 k-fold cross validation과 같은 개념이라고 보면 됨.
# mpiifacegaze 데이터셋은 15명의 사람이 있고, 각 사람마다 1개의 fold로 나누어서 훈련을 진행함.

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Gaze estimation using L2CSNet.')
    # Gaze360
    parser.add_argument(
        '--gaze360image_dir', dest='gaze360image_dir', help='Directory path for gaze images.',
        default='datasets/Gaze360/Image', type=str)
    parser.add_argument(
        '--gaze360label_dir', dest='gaze360label_dir', help='Directory path for gaze labels.',
        default='datasets/Gaze360/Label/train.label', type=str)
    # mpiigaze
    parser.add_argument(
        '--gazeMpiimage_dir', dest='gazeMpiimage_dir', help='Directory path for gaze images.',
        default='datasets/MPIIFaceGaze/Image', type=str)
    parser.add_argument(
        '--gazeMpiilabel_dir', dest='gazeMpiilabel_dir', help='Directory path for gaze labels.',
        default='datasets/MPIIFaceGaze/Label', type=str)

    # Important args -------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        '--dataset', dest='dataset', help='mpiigaze, rtgene, gaze360, ethgaze',
        default= "gaze360", type=str)
    parser.add_argument(
        '--output', dest='output', help='Path of output models.',
        default='output/snapshots/', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0] or multiple 0,1,2,3',
        default='0', type=str)
    parser.add_argument(
        '--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
        default=60, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=1, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], ''ResNet101, ResNet152, Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)
    parser.add_argument(
        '--alpha', dest='alpha', help='Regression loss coefficient.',
        default=1, type=float)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.00001, type=float)
    # ---------------------------------------------------------------------------------------------------------------------
    # Important args ------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    return args

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param
                
def get_ignored_params_mpii(model):
    # Generator function that yields ignored params.
    b = [model.module.conv1, model.module.bn1, model.module.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params_mpii(model):
    # Generator function that yields params that will be optimized.
    b = [model.module.layer1, model.module.layer2, model.module.layer3, model.module.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params_mpii(model):
    # Generator function that yields fc layer params.
    b = [model.module.fc_yaw_gaze, model.module.fc_pitch_gaze]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param
def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw_gaze, model.fc_pitch_gaze]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param
                
def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def getArch_weights(arch, bins): # backbone 모델을 가져오는 함수
    if arch == 'ResNet18':
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
        pre_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    elif arch == 'ResNet34':
        model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
    elif arch == 'ResNet101':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
    elif arch == 'ResNet152':
        model = L2CS(torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                  'The default value of ResNet50 will be used instead!')
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

    return model, pre_url

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = select_device(args.gpu_id, batch_size=args.batch_size)
    data_set=args.dataset
    alpha = args.alpha
    output=args.output
    
    # Data transformations
    # torchvision.transforms는 파이토치에서 이미지 데이터의 전처리 및 데이터 증강을 위해 제공하는 모듈입니다.    
    transformations = transforms.Compose([ # Compose: 여러 변환을 연달아 적용합니다.
        transforms.Resize(448), # Resize: 이미지의 크기를 조절합니다.
        transforms.ToTensor(), # ToTensor: 이미지를 텐서로 변환합니다.
        transforms.Normalize( # Normalize: 이미지를 정규화합니다.
            mean=[0.485, 0.456, 0.406], # 이미지의 평균인데, 이 값들은 ImageNet 데이터셋의 평균과 표준편차
            std=[0.229, 0.224, 0.225] # 이미지의 표준편차인데, 이 값들은 ImageNet 데이터셋의 평균과 표준편차
        )
    ])
    
    if data_set=="gaze360":
        model, pre_url = getArch_weights(args.arch, 90)
        if args.snapshot == '':
            load_filtered_state_dict(model, model_zoo.load_url(pre_url))
        else:
            saved_state_dict = torch.load(args.snapshot)
            model.load_state_dict(saved_state_dict)
        
        
        model.cuda(gpu)
        dataset=Gaze360(args.gaze360label_dir, args.gaze360image_dir, transformations, 180, 4)
        print('Loading data.')
        train_loader_gaze = DataLoader(
            dataset=dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=0,
            pin_memory=True)
        torch.backends.cudnn.benchmark = True

        summary_name = '{}_{}'.format('L2CS-gaze360-', int(time.time()))
        output=os.path.join(output, summary_name)
        if not os.path.exists(output):
            os.makedirs(output)

        
        criterion = nn.CrossEntropyLoss().cuda(gpu)
        reg_criterion = nn.MSELoss().cuda(gpu)
        softmax = nn.Softmax(dim=1).cuda(gpu)
        idx_tensor = [idx for idx in range(90)]
        idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)
        

        # Optimizer gaze
        optimizer_gaze = torch.optim.Adam([
            {'params': get_ignored_params(model), 'lr': 0},
            {'params': get_non_ignored_params(model), 'lr': args.lr},
            {'params': get_fc_params(model), 'lr': args.lr}
        ], args.lr)

        configuration = f"\ntrain configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\nStart testing dataset={data_set}, loader={len(train_loader_gaze)}------------------------- \n"
        print(configuration)
        for epoch in range(num_epochs):
            sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0

            
            for i, (images_gaze, labels_gaze, cont_labels_gaze,name) in enumerate(train_loader_gaze):
                images_gaze = Variable(images_gaze).cuda(gpu)
                
                # Binned labels
                label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
                label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)

                # Continuous labels
                label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
                label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

                pitch, yaw = model(images_gaze)

                # Cross entropy loss
                loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
                loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

                # MSE loss
                pitch_predicted = softmax(pitch)
                yaw_predicted = softmax(yaw)

                pitch_predicted = \
                    torch.sum(pitch_predicted * idx_tensor, 1) * 4 - 180
                yaw_predicted = \
                    torch.sum(yaw_predicted * idx_tensor, 1) * 4 - 180

                loss_reg_pitch = reg_criterion(
                    pitch_predicted, label_pitch_cont_gaze)
                loss_reg_yaw = reg_criterion(
                    yaw_predicted, label_yaw_cont_gaze)

                # Total loss
                loss_pitch_gaze += alpha * loss_reg_pitch
                loss_yaw_gaze += alpha * loss_reg_yaw

                sum_loss_pitch_gaze += loss_pitch_gaze
                sum_loss_yaw_gaze += loss_yaw_gaze

                loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
                grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
                optimizer_gaze.zero_grad(set_to_none=True)
                torch.autograd.backward(loss_seq, grad_seq)
                optimizer_gaze.step()
                # scheduler.step()
                
                iter_gaze += 1

                if (i+1) % 100 == 0:
                    print('Epoch [%d/%d], Iter [%d/%d] Losses: '
                        'Gaze Yaw %.4f,Gaze Pitch %.4f' % (
                            epoch+1,
                            num_epochs,
                            i+1,
                            len(dataset)//batch_size,
                            sum_loss_pitch_gaze/iter_gaze,
                            sum_loss_yaw_gaze/iter_gaze
                        )
                        )
                    
            if epoch % 1 == 0 and epoch < num_epochs:
                print('Taking snapshot...',
                    torch.save(model.state_dict(),
                                output +'/'+
                                '_epoch_' + str(epoch+1) + '.pkl')
                    )
            

    # mpiigaze
    elif data_set=="mpiigaze":
        folder = os.listdir(args.gazeMpiilabel_dir)
        folder.sort()
        testlabelpathombined = [os.path.join(args.gazeMpiilabel_dir, j) for j in folder]
        for fold in range(15): # 15개의 fold
            model, pre_url = getArch_weights(args.arch, 28)
            # 28개의 bin, bin이 뭐냐면 각도를 몇도로 나눌 것인지에 대한 것.
            # 왜 28이냐면 datasets.py를 보면 bins = np.array(range(-42, 42,3))이라고 되어 있음.
            # 즉, -42도부터 42도까지 3도 간격으로 나누면 28개가 됨. 왜 42냐고? 논문에서 사람 시야가 그 정도래.
            # model엔 L2CS 모델을, pre_url엔 backbone 모델(resnet)을 가져오는 것.
            
            load_filtered_state_dict(model, model_zoo.load_url(pre_url))
            # 이건 미리 학습된 모델을 불러오는 것. model_zoo.load_url은 미리 학습된 모델을 불러오는 함수.
            # 즉, resnet을 불러와서 L2CS 모델에 넣어줌.
            
            model = nn.DataParallel(model)
            # DataParallel은 여러 GPU를 사용하여 모델을 병렬로 실행할 수 있게 해주는 것.
            
            model.to(gpu)
            print('Loading data.')
            dataset=Mpiigaze(testlabelpathombined, args.gazeMpiimage_dir, transformations, True, 180, fold)
            # Mpiigaze는 데이터셋 클래스로, 데이터셋을 불러오는 역할을 함.
            # testlabelpathombined는 라벨이 있는 경로, args.gazeMpiimage_dir는 이미지가 있는 경로, transformations는 데이터 전처리를 위한 것.
            # True는 훈련 데이터셋을 불러오는 것이라는 것을 의미함.
            # fold는 leave-one-person-out을 위한 것.
            
            train_loader_gaze = DataLoader( # DataLoader는 데이터셋을 불러오는 역할을 함.
                dataset=dataset, 
                batch_size=int(batch_size),
                shuffle=True, # 데이터를 섞을 것인지에 대한 것.
                num_workers=4, # 데이터를 불러올 때 사용할 프로세스 수.
                pin_memory=True) # 데이터를 불러올 때 메모리를 고정할 것인지에 대한 것. 데이터를 고정하면 데이터를 불러올 때 빠르게 불러올 수 있음.
            torch.backends.cudnn.benchmark = True
            # cudnn은 NVIDIA에서 제공하는 딥러닝 라이브러리로, 딥러닝 모델을 빠르게 학습할 수 있도록 도와줌.
            # benchmark는 cudnn을 사용할 때 최적화를 위해 사용하는 것.
            
            summary_name = '{}_{}'.format('L2CS-mpiigaze', int(time.time()))

            if not os.path.exists(os.path.join(output+'/{}'.format(summary_name),'fold' + str(fold))):
                os.makedirs(os.path.join(output+'/{}'.format(summary_name),'fold' + str(fold)))

            criterion = nn.CrossEntropyLoss().cuda(gpu) # criterion은 손실 함수를 의미함.
            reg_criterion = nn.MSELoss().cuda(gpu) # reg_criterion은 회귀 손실 함수를 의미함.
            softmax = nn.Softmax(dim=1).cuda(gpu)
            idx_tensor = [idx for idx in range(28)] # idx_tensor는 각도를 나누는 것. 왜 28개냐면 28개의 bin이 있으니까.
            idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)
            # torch.FloatTensor는 torch.Tensor를 float형으로 변환하는 것.
            # torch.autograd.Variable은 역전파 연산을 위한 자동 미분을 지원하는 Tensor를 감싸는 클래스.
            # 지금은 deprecated되었지만, 여전히 사용 가능함.

            # Optimizer gaze
            optimizer_gaze = torch.optim.Adam([
                # {'params': get_ignored_params(model, args.arch), 'lr': 0},
                # {'params': get_non_ignored_params(model, args.arch), 'lr': args.lr},
                # {'params': get_fc_params(model, args.arch), 'lr': args.lr}
                {'params': get_ignored_params_mpii(model), 'lr': 0},
                {'params': get_non_ignored_params_mpii(model), 'lr': args.lr},
                {'params': get_fc_params_mpii(model), 'lr': args.lr}
            ], args.lr)
            # torch.optim.Adam은 Adam 최적화 알고리즘을 사용하는 것.
            # get_ignored_params, get_non_ignored_params, get_fc_params는 각각 무시할 파라미터, 최적화할 파라미터, fc 레이어 파라미터를 가져오는 것.
            # Adam 알고리즘은 쉽게 말하면 momentum과 RMSprop을 합친 것.
            # momentum은 관성을 의미하고, RMSprop은 학습률을 조절하는 것.
            
            configuration = f"\ntrain configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\n Start training dataset={data_set}, loader={len(train_loader_gaze)}, fold={fold}--------------\n"
            print(configuration)
            for epoch in range(num_epochs): # epoch만큼 반복
                sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0
                # sum_loss_pitch_gaze, sum_loss_yaw_gaze는 각각 pitch와 yaw의 손실을 의미함.
                # iter_gaze는 반복 횟수를 의미함. 나중에 평균 loss 계산에 사용함.
                
                # train_loader_gaze에서 데이터를 하나씩 배치 단위로 가져와 학습하는 루프
                for i, (images_gaze, labels_gaze, cont_labels_gaze, name) in enumerate(train_loader_gaze):
                    # train_loader로 커스텀된 데이터를 가져왔음.
                    # image_gaze에는 이미지 자체가, labels_gaze에는 도 단위의 gaze2d정보를 라디안으로 변경한 뒤 bin에 넣은 값, 
                    # cont_labels_gaze에는 도 단위의 gaze2d정보를 라디안으로 변경한 값, name에는 이미지 이름이 대응되어서 가져와진 것
                    
                    images_gaze = Variable(images_gaze).cuda(gpu)
                    # image_gaze는 입력 이미지 데이터를 의미하며, gpu에 이미지를 올려서 모델이 처리할 수 있도록 함

                    # Binned labels
                    # pitch와 yaw가 어느 bin에 속하는지에 대한 라벨
                    # labels_gaze[:, 0] 은 numpy배열의 모든 행 중, 0번째 열의 값들을 가져오겠다는 뜻. 
                    # 여기선 labels_gaze가 1차원 배열(vector)이라 그냥 벡터 중 0번째 값을 가져왔다고 보면 됨.
                    label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
                    label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)

                    # Continuous labels
                    # pitch와 yaw의 실제 각도
                    label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
                    label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

                    # 모델에 이미지를 넣어서 pitch와 yaw를 예측함.(forward pass)
                    pitch, yaw = model(images_gaze)

                    # Cross entropy loss
                    # 각도 bin에 대한 분류 예측에 대해 CrossEntropyLoss 계산
                    loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
                    loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

                    # MSE loss
                    # softmax를 통해 각 bin에 대한 확률 분포 획득
                    pitch_predicted = softmax(pitch)
                    yaw_predicted = softmax(yaw)

                    # 기대값 계산: softmax확률 * bin index -> 실제 각도(degrees)로 변환
                    pitch_predicted = \
                        torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
                    yaw_predicted = \
                        torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42
                    # *3은 bin 하나가 3도 간격이고, -42는 bin의 시작 각도가 -42도이기 때문

                    # 예측 continuous 각도와 실제 각도 사이의 회귀 손실(MSELoss) 계산
                    loss_reg_pitch = reg_criterion(
                        pitch_predicted, label_pitch_cont_gaze)
                    loss_reg_yaw = reg_criterion(
                        yaw_predicted, label_yaw_cont_gaze)

                    # Total loss
                    # 총 손실 = 분류 손실 + a * 회귀 손실(회귀 손실에 대한 가중치 a)
                    loss_pitch_gaze += alpha * loss_reg_pitch
                    loss_yaw_gaze += alpha * loss_reg_yaw

                    # 누적 손실
                    sum_loss_pitch_gaze += loss_pitch_gaze
                    sum_loss_yaw_gaze += loss_yaw_gaze

                    # 두 손실을 하나의 리스트로 묶어서 역전파
                    loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
                    grad_seq = \
                        [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
                    # grad_seq는 각 손실에 gradient 비율(기본적으로 동일하게 1)

                    # 기존 gradient 초기화
                    optimizer_gaze.zero_grad(set_to_none=True)
                    
                    # 역전파
                    torch.autograd.backward(loss_seq, grad_seq)
                    
                    # 가중치 업데이트
                    optimizer_gaze.step()

                    # 이터레이션 수 증가가
                    iter_gaze += 1

                    # 매 100번째 이터레이션마다 현재까지의 평균 손실 출력
                    if (i+1) % 100 == 0:
                        print('Epoch [%d/%d], Iter [%d/%d] Losses: '
                            'Gaze Yaw %.4f,Gaze Pitch %.4f' % (
                                epoch+1,
                                num_epochs,
                                i+1,
                                len(dataset)//batch_size,
                                sum_loss_pitch_gaze/iter_gaze,
                                sum_loss_yaw_gaze/iter_gaze
                            )
                        )

                # Save models at numbered epochs.
                if epoch % 1 == 0 and epoch < num_epochs:
                    print('Taking snapshot...',
                        torch.save(model.state_dict(),
                                    # output+'/fold' + str(fold) +'/'+
                                    output+'fold' + str(fold) +'/'+
                                    '_epoch_' + str(epoch+1) + '.pkl')
                        )
                    
                    # 만약 온디바이스 추론을 위한 빠른 모델 로딩이 필요하다면 onnx를 사용할 수 있음.
                    # torch.onnx.export(model, images_gaze, output+'/fold' + str(fold) +'/'+
                    #                   '_epoch_' + str(epoch+1) + '.onnx')
