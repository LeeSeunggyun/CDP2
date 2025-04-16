import torch
import os
import torch
import os
import argparse
from l2cs import L2CS
import torchvision

def parse_args():
    parser = argparse.ArgumentParser(description='Quantize L2CSNet model.')
    parser.add_argument(
        '--dataset', dest='dataset', help='gaze360, mpiigaze',
        default= "gaze360", type=str)
    parser.add_argument('--snapshot', type=str, required=True, help='Path to folder with .pkl model files')
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], ''ResNet101, ResNet152, Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)
    parser.add_argument('--bins', type=int, default=90, help='Number of bins (e.g. 90 for Gaze360, 28 for MPIIGaze)')
    return parser.parse_args()

def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model

def main():
    args = parse_args()

    snapshot_folder = args.snapshot
    arch = args.arch
    bins = args.bins

    model_files = sorted([f for f in os.listdir(snapshot_folder) if f.endswith('.pkl')])

    if not model_files:
        print(f"No .pkl files found in {snapshot_folder}")
        return

    print(f"Found {len(model_files)} model(s) in {snapshot_folder}. Starting quantization...")

    for file in model_files:
        print(f"Quantizing {file}...")

        # 모델 초기화 및 state_dict 로드
        model = getArch(arch, bins)
        model.eval()
        state_dict = torch.load(os.path.join(snapshot_folder, file), map_location='cpu')
        model.load_state_dict(state_dict)

        # 양자화 적용
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

        # 저장 경로
        filename_wo_ext = os.path.splitext(file)[0]
        save_path = os.path.join(snapshot_folder, f"quantized_{filename_wo_ext}.pkl")

        # quantization.py 안에서
        torch.save(quantized_model, save_path)  # 전체 모델 저장

        print(f"Saved quantized model to {save_path}")

if __name__ == '__main__':
    main()
