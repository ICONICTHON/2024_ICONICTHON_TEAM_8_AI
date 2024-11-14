import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import requests

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # CORS 허용

# YOLOv8 모델 로드
MODEL_PATH = os.path.join("models", "The Best.pt")  # 상대 경로 사용
model = YOLO(MODEL_PATH)

# 탐지 상태 저장
detection_counters = {}  # 탐지 횟수 저장
CAPTURE_CLASSES = {"card", "wallets"}  # 캡처 대상 클래스

# 백엔드 서버 URL
BACKEND_API_URL = 'https://all-laf.duckdns.org/'  # Express 서버의 주소로 변경

@app.route('/detect', methods=['POST'])
def detect():
    # 이미지 데이터 수신
    if 'image' not in request.files:
        return jsonify({"error": "이미지가 업로드되지 않았습니다."}), 400

    file = request.files['image']
    if not file:
        return jsonify({"error": "이미지가 업로드되지 않았습니다."}), 400

    # 이미지 읽기
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # 모델 추론
    results = model(img, conf=0.3)  # 신뢰도 0.5 이상 탐지
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes.data is not None else []

    # 탐지된 클래스와 신뢰도
    detected_classes = []
    height, width, _ = img.shape
    for box in detections:
        x1, y1, x2, y2, confidence, cls = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)  # 클래스 ID
        confidence = float(confidence)  # 신뢰도
        class_name = model.names[cls]

        # 탐지된 클래스 저장
        detected_classes.append({"class": class_name, "confidence": confidence})

        if confidence >= 0.4 and class_name in CAPTURE_CLASSES:
            # 탐지 영역에 10픽셀 여유 추가
            x1 = max(0, x1 - 10)
            y1 = max(0, y1 - 10)
            x2 = min(width, x2 + 10)
            y2 = min(height, y2 + 10)

            # 캡처된 이미지를 저장
            cropped_img = img[y1:y2, x1:x2]
            _, buffer = cv2.imencode('.jpg', cropped_img)

            # 탐지 횟수 증가
            detection_counters[class_name] = detection_counters.get(class_name, 0) + 1

            if detection_counters[class_name] >= 2:
                # 캡처된 이미지를 백엔드로 전송
                try:
                    backend_response = requests.post(
                        f"{BACKEND_API_URL}/save-image",
                        files={"image": ('capture.jpg', buffer.tobytes(), 'image/jpeg')}
                    )
                    print("캡처된 이미지 전송 완료:", backend_response.json())
                except Exception as e:
                    print("캡처된 이미지 전송 실패:", str(e))

                # 탐지 횟수 초기화
                detection_counters[class_name] = 0
        else:
            # 탐지되지 않으면 횟수 초기화
            detection_counters[class_name] = 0

    # 탐지된 클래스 반환
    return jsonify({
        "status": "탐지 성공" if detected_classes else "탐지 실패",
        "detections": detected_classes
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Koyeb에서 제공하는 PORT 환경 변수
    app.run(host="0.0.0.0", port=port)
