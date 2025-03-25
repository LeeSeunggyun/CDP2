from l2cs import Pipeline, render
import cv2
import torch
import time

# 시선 추정 파이프라인 초기화
gaze_pipeline = Pipeline(
    weights="./models/L2CSNet_gaze360.pkl",
    # weights = "./models/MPIIGaze/fold7.pkl",
    arch='ResNet50',
    device=torch.device('cpu')  # GPU 사용 시 'cuda'로 변경
)

# 웹캠에서 영상 스트림 가져오기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# FPS 계산을 위한 변수 초기화
prev_time = time.time()
frame_count = 0
fps = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 현재 시간 측정 (시작)
        start_time = time.time()

        # 시선 추정 수행
        results = gaze_pipeline.step(frame)

        # 시각화
        output_frame = render(frame, results)

        # FPS 계산
        frame_count += 1
        elapsed_time = time.time() - prev_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            prev_time = time.time()
            frame_count = 0

        # FPS를 이미지에 표시
        cv2.putText(output_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 결과 출력
        cv2.imshow('L2CS Gaze Estimation - Live', output_frame)

        # 'q' 키 입력 시 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
