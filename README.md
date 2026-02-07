import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ----------------------------------------------------------------
# 성능 최적화 설정
# model_complexity=0 : 성능 위주 (Lite 모델) / 1 : 기본 / 2 : 정확도 위주
# ----------------------------------------------------------------
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0) # 0으로 설정하여 속도 향상

cap = cv2.VideoCapture(0)

# (선택) 해상도를 줄이면 FPS가 더 올라갑니다. 필요 시 주석 해제하세요.
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("종료하려면 'q' 키를 누르세요.")

prev_time = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("카메라를 찾을 수 없습니다.")
        continue

    image.flags.writeable = False
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = hands.process(image_rgb)

    image.flags.writeable = True
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(248, 144, 0), thickness=3, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=1)
            )

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    image = cv2.flip(image, 1)
    cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Optimized Hand Detection', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
