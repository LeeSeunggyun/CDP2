# L2CS-Net
-cuda를 쓸 수 있는 환경(nvidia gpu)이 못 되어서 훈련을 직접 해보진 못함. Prepare datasets 부터 보면 됨.
-conda 가상환경에서 작업함.

## Usage

Detect face and predict gaze from webcam

```python
from l2cs import Pipeline, render
import cv2

gaze_pipeline = Pipeline(
    weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
    arch='ResNet50',
    device=torch.device('cpu') # or 'gpu'
)
 
cap = cv2.VideoCapture(cam)
_, frame = cap.read()    

# Process frame and visualize
results = gaze_pipeline.step(frame)
frame = render(frame, results)
```

## Demo
* Download the pre-trained models from [here](https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd?usp=sharing) and Store it to *models/*.
*  Run:
```
 python demo.py \
 --snapshot models/L2CSNet_gaze360.pkl \
 --gpu 0 \
 --cam 0 \
```
This means the demo will run using *L2CSNet_gaze360.pkl* pretrained model

## Community Contributions

- [Gaze Detection and Eye Tracking: A How-To Guide](https://blog.roboflow.com/gaze-direction-position/): Use L2CS-Net through a HTTP interface with the open source Roboflow Inference project.

## MPIIGaze
We provide the code for train and test MPIIGaze dataset with leave-one-person-out evaluation.

### Prepare datasets
* download datasets from Notion
* Store the dataset to *datasets/MPIIFaceGaze*.
* It should be like
```
├─datasets/
│  └─MPIIFaceGaze/
│      ├─Image/
│      │  ├─p00/
│      │  ├─...
│      │  └─p14/
│      └─Label/
├─l2cs/
├─models/
├─output/
│    └─snapshots/
└─other files
```

### Install requirements.txt
```
 pip install -r requirements.txt
```

### Train
```
 python train.py \
 --dataset mpiigaze \
 --snapshot output/snapshots \
 --gpu 0 \
 --num_epochs 50 \
 --batch_size 16 \
 --lr 0.00001 \
 --alpha 1 \

```
This means the code will perform leave-one-person-out training automatically and store the models to *output/snapshots*.

### Test
```
 python test.py \
 --dataset mpiigaze \
 --snapshot output/snapshots/snapshot_folder \
 --evalpath evaluation/L2CS-mpiigaze  \
 --gpu 0 \
```
This means the code will perform leave-one-person-out testing automatically and store the results to *evaluation/L2CS-mpiigaze*.

To get the average leave-one-person-out accuracy use:
```
 python leave_one_out_eval.py \
 --evalpath evaluation/L2CS-mpiigaze  \
 --respath evaluation/L2CS-mpiigaze  \
```
This means the code will take the evaluation path and outputs the leave-one-out gaze accuracy to the *evaluation/L2CS-mpiigaze*.
