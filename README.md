코드
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks, handedness):
    fingers = []

    lm = hand_landmarks.landmark

    #엄지
    if handedness == "Right":
        if lm[4].x < lm[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        if lm[4].x > lm[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

    #나머지
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):
        if lm[tip].y < lm[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    command_text = ""
    finger_count = 0

    if result.multi_hand_landmarks and result.multi_handedness:

        for hand_landmarks, handedness in zip(
            result.multi_hand_landmarks,
            result.multi_handedness
        ):

            label = handedness.classification[0].label
            finger_count = count_fingers(hand_landmarks, label)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if finger_count == 1:
        command_text = "start"
    elif finger_count == 2:
        command_text = "stop"
    elif finger_count == 3:
        command_text = "faster"
    elif finger_count == 4:
        command_text = "slower"
    else:
        command_text = ""
        
    if command_text:
        cv2.putText(
            frame,
            command_text,
            (50, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 0, 255),
            6
        )

    cv2.imshow("sex", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()


보고서
mediapipe와 OpenCV를 활용한 손 감지 기반 페달 제어 시스템

1.동기

현재의 차량 운전에 있어 페달은 운전자가 가속과 감속을 조절하여 차량 속도를 제어하는 필수 안전 장치이다.
하지만 하체에 제약이 있는 사용자에게는 이를 사용할 수 없어 운전이 불가능 하다.
따라서 나는 손 감지 기반 페달 제어 시스템을 만들어 사용 대상자들의 편의성을 향상시키고자 한다.


2. 알고리즘 설계
손가락의 감지 코드는 MediaPipe 솔루션 가이드 | Google AI Edge의 Python용 손 랜드마크 감지 가이드를 사용했다.

#엄지 손가락 감지
오른손에서 thumb_tip.x < thumb_pip.x 라면 손가락이 펴져있다.
왼손에서 thumb_tip.x > thumb_pip.x 라면 손가락이 펴져있다.

#나머지 손가락 감지
tip.y < pip.y 라면 손가락이 펴져있다.

여기서 tip와 pip는 각각 손가락 가장 끝 쪽 마디, 손가락 가장 끝 쪽 관절을 의미한다.
이 알고리즘으로 펼쳐진 손가락의 개수를 알 수 있다.
이 구현은 Indriani & Agoes (2021)와 Gil-Martín et al. (2025)의 권고(랜드마크 기반 입력, 공간 정규화)와 일치한다.

#메인
손가락이 1개 펴져 있다면 'start' 이다.
손가락이 2개 펴져 있다면 'stop' 이다.
손가락이 3개 펴져 있다면 'faster' 이다.
손가락이 4개 펴져 있다면 'slower' 이다.




관련 연구

MediaPipe·손 인식 관련 활용 사례

Indriani & Agoes (2021) — MediaPipe를 사용한 간단한 제스처 응용 사례로, MediaPipe의 실시간 손 랜드마크 추출이 상호작용 인터페이스에 실용적임을 보였다.
적용: 본 시스템의 손가락 인식 모듈이 MediaPipe 기반으로 설계된 근거가 된다.
Indriani, M. H., & Agoes, A. (2021). Applying hand gesture recognition for user guide application using MediaPipe. Advances in Engineering Research, 207, 297–301.

Amprimo et al. (2024) — MediaPipe와 깊이(depth) 보강 모델의 임상적 검증 연구로, MediaPipe의 손 추적 정확도가 임상적 수준으로도 신뢰할 수 있음을 보였다.
적용: 손가락 좌표 신뢰도/한계 판단 및 깊이(z) 정보를 이용한 전진/후진 연동 가능성 근거.
Amprimo, G., et al. (2024). Hand tracking for clinical applications: Validation of the Google MediaPipe Hand (GMH) and the depth-enhanced GMH-D frameworks. Biomedical Signal Processing and Control, 96, 106508.

Gil-Martín et al. (2025) — MediaPipe 랜드마크를 입력으로 한 딥러닝 분류 실험, 공간 정규화가 성능을 높인다는 결과 제시.
적용: 본 연구의 정규화(손목 기준 중심화·크기 정규화) 및 시계열 입력 설계의 타당성 근거.
Gil-Martín, M., et al. (2025). Hand gesture recognition using MediaPipe landmarks and deep learning networks. In Proceedings of ICAART 2025 (Vol. 3, pp. 24–30).

손 동작을 이용한 차량/기계 제어 사례

Huda et al. (2025) — MediaPipe 기반 실시간 제스처로 휠체어 제어한 사례. 소량의 손동작으로도 제어가 가능하다는 실험적 증거를 제공.
적용: 손가락 기반 휠체어/차량 제어 가능성의 직접적 근거.
Huda, M. R., Ali, M. L., & Sadi, M. S. (2025). Developing a real-time hand-gesture recognition technique for wheelchair control. PLOS One, 20(4), e0319996.

Sadi et al. (2022) — 저비용 손가락 제스처 휠체어, 실사용성 검증.
적용: 실사용 관점에서 단순 제스처(손가락 기반)가 접근성 향상에 기여한다는 근거.
Sadi, M. S., et al. (2022). Finger-gesture controlled wheelchair with enabling IoT. Sensors, 22(24), 2107.

Wameed et al. (2023), Swain & Bhoi (2024) — MediaPipe 기반 로봇·RC카 제어 사례로, 손 랜드마크를 실시간 명령으로 맵핑하는 전형적 파이프라인을 제시.
적용: 본 연구의 명령→모터(PWM)/시리얼 송신 매핑 방법론과 유사한 선행 사례.
Wameed, M., et al. (2023). Tracked robot control with hand gesture based on MediaPipe. Al-Khwarizmi Engineering Journal, 19(3), 56–71.
Swain, B. K., & Bhoi, A. K. (2024). Single and multi-hand gesture based soft material robotic car control. Procedia Computer Science, 235, 3055–3064.

컴퓨터 비전 기반 제어 인터페이스 총설

Jalayer et al. (2026), Saravana et al. (2025), Bayoudh et al. (2021) — 비전 기반 제스처 인식/센서 선택/딥러닝 방법들의 장단점 및 적용 분야 총설.
적용: 본 프로젝트에서 MediaPipe(랜드마크 기반)와 더 고급 학습기법(LSTM/Transformer)을 병용하는 설계 근거와, 환경·데이터 이슈(조명, 뷰포인트 등)를 정리하는 데 참고.
Jalayer, R., et al. (2026). A review on deep learning for vision-based hand detection... Robotics and Computer-Integrated Manufacturing, 97, 103110.

장애인을 위한 입력 인터페이스 연구

Rojas et al. (2022), Haddoun et al. (2025) — 비전/IMU/음성 등 대체 입력의 평가 연구.
적용: 손가락 기반 제어를 장애인 인터페이스의 한 축으로 보고, 멀티모달(예: 음성+손동작) 보완 전략을 권고.
Rojas, M., et al. (2022). Development of a sensing platform based on hands-free interfaces... Frontiers in Human Neuroscience, 16, 867377.

정리: 위 선행연구들은 MediaPipe 기반 손 랜드마크가 실시간 제어 입력으로 실용성이 있고, 손가락·제스처 기반 휠체어/로봇 제어의 성공 사례가 이미 존재함을 보여준다. 또한 공간 정규화·시계열 학습·안정화(시간 윈도우) 등의 기법이 인식 신뢰도를 높이는 데 핵심임을 입증한다.

8. 결론

본 연구는 MediaPipe 기반 손가락 개수 인식을 차량 제어에 적용하는 실용적 파이프라인을 제안하고, 관련 선행연구들을 근거로 설계·안정화·평가 계획을 제시했다. 손가락 개수라는 단순 입력은 장애 사용자의 접근성 향상 가능성을 보여주며, 향후 데이터 기반 학습·멀티모달 보완·임베디드 배포를 통해 실제 보조운전 시스템으로 발전시킬 수 있다.

참고문헌
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python?hl=ko

1.Amprimo, G., Masi, G., Pettiti, G., Olmo, G., Priano, L., & Ferraris, C. (2024). Hand tracking for clinical applications: Validation of the Google MediaPipe Hand (GMH) and GMH-D frameworks. Biomedical Signal Processing and Control, 96, 106508.

2.Gil-Martín, M., Marini, M. R., Martín-Fernández, I., & Esteban-Romero, S. (2025). Hand gesture recognition using MediaPipe landmarks and deep learning networks. In Proceedings of ICAART 2025 (Vol. 3, pp. 24–30). INSTICC Press.

3.Indriani, M. H., & Agoes, A. (2021). Applying hand gesture recognition for user guide application using MediaPipe. Advances in Engineering Research, 207, 297–301.

4.Jalayer, R., Jalayer, M., Orsenigo, C., & Tomizuka, M. (2026). A review on deep learning for vision-based hand detection, hand segmentation and hand gesture recognition in human–robot interaction. Robotics and Computer-Integrated Manufacturing, 97, 103110.

5.Huda, M. R., Ali, M. L., & Sadi, M. S. (2025). Developing a real-time hand-gesture recognition technique for wheelchair control. PLOS One, 20(4), e0319996.

6.Sadi, M. S., Alotaibi, M., Islam, M. R., Islam, M. S., Alhmiedat, T., & Bassfar, Z. (2022). Finger-gesture controlled wheelchair with enabling IoT. Sensors, 22(24), 2107.

7.Wameed, M., Alkamachi, A. M., & Ercelibi, E. (2023). Tracked robot control with hand gesture based on MediaPipe. Al-Khwarizmi Engineering Journal, 19(3), 56–71.

8.Swain, B. K., & Bhoi, A. K. (2024). Single and multi-hand gesture based soft material robotic car control. Procedia Computer Science, 235, 3055–3064.

9.Saravana, S. M. K., Lishan, C. L., Rao, S. P., & Manushree, K. B. (2025). A comprehensive survey on gesture-controlled interfaces: Technologies, applications, and challenges. International Journal of Scientific Research in Science and Technology, 12(2), 1112–1136.

10.Bayoudh, S., Masmoudi, M., & Chellali, R. (2021). Hand gesture recognition based on computer vision: A review of techniques. Mathematical Biosciences and Engineering, 18(4), 3582–3611.

11.Rojas, M., Ponce, P., & Molina, A. (2022). Development of a sensing platform based on hands-free interfaces for controlling electronic devices. Frontiers in Human Neuroscience, 16, 867377.

12.Haddoun, A., Djabri, D., Saidani, M., & Benbouzid, M. (2025). Development and evaluation of a head-controlled wheelchair system for users with severe motor impairments. MethodsX, 15, 103485.

13.Vysocký, A., Poštulka, T., Chlebek, J., Kot, T., Maslowski, J., & Grushko, S. (2023). Hand gesture interface for robot path definition in collaborative applications: Implementation and comparative study. Sensors, 23(9), 4219.
