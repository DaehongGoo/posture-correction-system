from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import math
import numpy as np
import time
import os

app = Flask(__name__)
cap = cv2.VideoCapture(0)

# 캡처 저장 폴더
if not os.path.exists('static/captures'):
    os.makedirs('static/captures')

# --- 상태 변수 ---
# status: "stopped" | "awaiting" | "running" | "paused"
status = "stopped"
start_time = None
total_time, safe_time, warning_time, danger_time = 0, 0, 0, 0
baseline_neck, baseline_shoulder = None, None
pitch_px, shoulder_diff, roll_diff, fwd_ratio, trunk_angle = 0, 0, 0, 0, 0

# --- 좌표 저장 ---
current_neck_center = None
current_shoulder_center = None

# --- baseline 윤곽(실루엣) ---
baseline_contours = None  # list of contours
baseline_alpha = 0.4      # 반투명도

# --- 최악 순간 캡처/수치 ---
max_pitch_px, max_shoulder_diff, max_roll_diff, max_fwd_ratio, max_trunk_angle = 0, 0, 0, 0, 0
pitch_capture, shoulder_capture, roll_capture, fwd_capture, trunk_capture = None, None, None, None, None
# --- 수정 ---: 최악의 순간 발생 시간 저장 변수 추가
pitch_capture_time, shoulder_capture_time, roll_capture_time, fwd_capture_time, trunk_capture_time = 0, 0, 0, 0, 0


def get_angle(p1, p2, vertical=False):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return abs(90 - abs(angle)) if vertical else abs(angle)


def angle_between_vectors(p1, p2, q1, q2):
    v1 = np.array([p2[0]-p1[0], p2[1]-p1[1]])
    v2 = np.array([q2[0]-q1[0], q2[1]-q1[1]])
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return math.degrees(math.acos(np.clip(dot / norm, -1.0, 1.0))) if norm else 0


def all_metrics_normal():
    """모든 항목이 정상 구간인지 검사(하한 기준)"""
    return (pitch_px < 15 and
            shoulder_diff < 10 and
            roll_diff < 4 and
            fwd_ratio < 1.5 and
            trunk_angle < 5)


def gen_frames():
    global start_time, total_time, safe_time, warning_time, danger_time
    global baseline_neck, baseline_shoulder
    global pitch_px, shoulder_diff, roll_diff, fwd_ratio, trunk_angle
    global max_pitch_px, max_shoulder_diff, max_roll_diff, max_fwd_ratio, max_trunk_angle
    global pitch_capture, shoulder_capture, roll_capture, fwd_capture, trunk_capture
    global current_neck_center, current_shoulder_center
    global baseline_contours, status
    # --- 수정 ---: 시간 변수 global에 추가
    global pitch_capture_time, shoulder_capture_time, roll_capture_time, fwd_capture_time, trunk_capture_time

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    segmentor = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    while True:
        success, frame = cap.read()
        if not success:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        segmentation = segmentor.process(rgb)

        now = time.time()
        delta = 0

        # 시간 누적은 오직 "running"일 때만
        if status == "running":
            if start_time is None:
                start_time = now
            else:
                delta = now - start_time
                total_time += delta
                start_time = now
        else:
            start_time = None

        silhouette_color = (0, 255, 0)  # default green

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            def px(idx): return int(lm[idx].x * w), int(lm[idx].y * h)
            l_ear, r_ear = px(7), px(8)
            l_sh, r_sh = px(11), px(12)
            nose = px(0)

            ear_center = ((l_ear[0]+r_ear[0])//2, (l_ear[1]+r_ear[1])//2)
            shoulder_center = ((l_sh[0]+r_sh[0])//2, (l_sh[1]+r_sh[1])//2)
            neck_center = (ear_center[0], ear_center[1]+15)

            current_neck_center = neck_center
            current_shoulder_center = shoulder_center

            # 5가지 지표
            pitch_px = abs(nose[1] - ear_center[1])
            shoulder_diff = abs(l_sh[1] - r_sh[1])
            roll_diff = abs(get_angle(l_ear, l_sh, True) - get_angle(r_ear, r_sh, True))
            ear_x, sh_x = (l_ear[0] + r_ear[0]) / 2, (l_sh[0] + r_sh[0]) / 2
            fwd_ratio = abs(ear_x - sh_x) / w * 100  # %

            if baseline_neck and baseline_shoulder:
                trunk_angle = angle_between_vectors(baseline_neck, baseline_shoulder, neck_center, shoulder_center)
            else:
                trunk_angle = 0

            # --- 자동 baseline 설정 (awaiting → running) ---
            if status == "awaiting" and segmentation.segmentation_mask is not None:
                if all_metrics_normal():
                    baseline_neck = neck_center
                    baseline_shoulder = shoulder_center
                    mask = (segmentation.segmentation_mask > 0.5).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    baseline_contours = contours if contours else None
                    status = "running"
                    start_time = now

            # --- 최악 캡처 ---
            if status == "running":
                # --- 수정 ---: 최악의 순간 갱신 시, 'total_time'도 함께 저장
                if pitch_px > max_pitch_px:
                    max_pitch_px = pitch_px; pitch_capture = frame.copy(); pitch_capture_time = total_time
                if shoulder_diff > max_shoulder_diff:
                    max_shoulder_diff = shoulder_diff; shoulder_capture = frame.copy(); shoulder_capture_time = total_time
                if roll_diff > max_roll_diff:
                    max_roll_diff = roll_diff; roll_capture = frame.copy(); roll_capture_time = total_time
                if fwd_ratio > max_fwd_ratio:
                    max_fwd_ratio = fwd_ratio; fwd_capture = frame.copy(); fwd_capture_time = total_time
                if trunk_angle > max_trunk_angle:
                    max_trunk_angle = trunk_angle; trunk_capture = frame.copy(); trunk_capture_time = total_time

            # --- 구간 누적 & 실루엣 색상 ---
            if status == "running":
                if (pitch_px > 25 or shoulder_diff > 20 or roll_diff > 8 or fwd_ratio > 3 or trunk_angle > 10):
                    danger_time += delta; silhouette_color = (0, 0, 255)
                elif (pitch_px > 15 or shoulder_diff > 10 or roll_diff > 4 or fwd_ratio > 1.5 or trunk_angle > 5):
                    warning_time += delta; silhouette_color = (0, 165, 255)
                else:
                    safe_time += delta; silhouette_color = (0, 255, 0)

            # --- 현재 실루엣 ---
            if segmentation.segmentation_mask is not None:
                mask = (segmentation.segmentation_mask > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    cv2.drawContours(frame, [contour], -1, silhouette_color, 2)
        else:
            current_neck_center = None
            current_shoulder_center = None

        # --- baseline 윤곽(반투명 오버레이) ---
        if baseline_contours is not None:
            overlay = frame.copy()
            for contour in baseline_contours:
                cv2.drawContours(overlay, [contour], -1, (255, 255, 255), 2)
            cv2.addWeighted(overlay, baseline_alpha, frame, 1 - baseline_alpha, 0, frame)

        # --- 정지 상태 십자선 ---
        if status == "stopped":
            center_x, center_y = w // 2, h // 2
            cv2.line(frame, (center_x, 0), (center_x, h), (255, 255, 255), 1)
            cv2.line(frame, (0, center_y), (w, center_y), (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# --- 유틸 초기화 ---
def reset_detection_state():
    """실시간 좌표만 초기화"""
    global current_neck_center, current_shoulder_center
    current_neck_center = None
    current_shoulder_center = None


def reset_home_state():
    """
    홈(index) 진입 시 화면을 '완전 초기' 상태로 보이게끔 하는 초기화.
    """
    global baseline_neck, baseline_shoulder, baseline_contours
    global status, start_time
    baseline_neck = None
    baseline_shoulder = None
    baseline_contours = None
    status = "stopped"
    start_time = None


def reset_all():
    """새 세션 시작 전 전체 리셋"""
    global total_time, safe_time, warning_time, danger_time, start_time
    global max_pitch_px, max_shoulder_diff, max_roll_diff, max_fwd_ratio, max_trunk_angle
    global pitch_capture, shoulder_capture, roll_capture, fwd_capture, trunk_capture
    global baseline_neck, baseline_shoulder, baseline_contours
    # --- 수정 ---: 시간 변수 초기화 추가
    global pitch_capture_time, shoulder_capture_time, roll_capture_time, fwd_capture_time, trunk_capture_time

    total_time = safe_time = warning_time = danger_time = 0
    start_time = None
    max_pitch_px = max_shoulder_diff = max_roll_diff = max_fwd_ratio = max_trunk_angle = 0
    pitch_capture = shoulder_capture = roll_capture = fwd_capture = trunk_capture = None
    baseline_neck = baseline_shoulder = None
    baseline_contours = None
    # --- 수정 ---
    pitch_capture_time = shoulder_capture_time = roll_capture_time = fwd_capture_time = trunk_capture_time = 0


# --- 라우트 ---
@app.route('/')
def index():
    reset_detection_state()
    reset_home_state()
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start')
def start():
    """
    - stopped → awaiting (자동 초기자세 대기)
    - paused  → running  (재시작)
    """
    global status, start_time
    if status == "stopped":
        reset_all()
        status = "awaiting"
        start_time = None
        return "Awaiting Baseline"
    elif status == "paused":
        status = "running"
        start_time = time.time()
        return "Resumed"
    else:
        return "Already Started"


@app.route('/stop')
def stop():
    global status, start_time
    if status in ["running", "awaiting"]:
        status = "paused"
    start_time = None
    return "Paused"


@app.route('/end')
def end():
    global status
    status = "stopped"

    if pitch_capture is not None: cv2.imwrite('static/captures/pitch.jpg', pitch_capture)
    if shoulder_capture is not None: cv2.imwrite('static/captures/shoulder.jpg', shoulder_capture)
    if roll_capture is not None: cv2.imwrite('static/captures/roll.jpg', roll_capture)
    if fwd_capture is not None: cv2.imwrite('static/captures/fwd.jpg', fwd_capture)
    if trunk_capture is not None: cv2.imwrite('static/captures/trunk.jpg', trunk_capture)
    return "Ended and Captures Saved"


@app.route('/stats')
def stats():
    """실시간 측정 데이터 + 상태를 JSON으로 반환 (index에서 사용)"""
    def status_text(value, thresholds):
        if value < thresholds[0]: return ('정상', 'green')
        elif value < thresholds[1]: return ('경고', 'orange')
        else: return ('비정상', 'red')

    pitch_stat, pitch_color = status_text(pitch_px, [15, 25])
    shoulder_stat, shoulder_color = status_text(shoulder_diff, [10, 20])
    roll_stat, roll_color = status_text(roll_diff, [4, 8])
    fwd_stat, fwd_color = status_text(fwd_ratio, [1.5, 3])
    trunk_stat, trunk_color = status_text(trunk_angle, [5, 10])

    return jsonify({
        '고개숙임': {'value': f'{pitch_px:.1f}px', 'status': pitch_stat, 'color': pitch_color},
        '어깨높이차': {'value': f'{shoulder_diff:.1f}px', 'status': shoulder_stat, 'color': shoulder_color},
        '고개좌우기울기': {'value': f'{roll_diff:.1f}도', 'status': roll_stat, 'color': roll_color},
        '어깨말림': {'value': f'{fwd_ratio:.1f}%', 'status': fwd_stat, 'color': fwd_color},
        '상체좌우기울기': {'value': f'{trunk_angle:.1f}도', 'status': trunk_stat, 'color': trunk_color},
        '전체측정시간': f'{total_time:.1f}초',
        '올바른자세시간': f'{safe_time:.1f}초',
        '경고자세시간': f'{warning_time:.1f}초',
        '위험자세시간': f'{danger_time:.1f}초',
        'phase': status
    })


@app.route('/results')
def results():
    return render_template('results.html')


@app.route('/results_data')
def results_data():
    """결과 페이지용: 시간 + 각 항목의 최악 상태 라벨 + 발생 시간 반환"""
    def status_label(value, thresholds):
        if value < thresholds[0]: return "정상"
        elif value < thresholds[1]: return "경고"
        else: return "위험"

    return jsonify({
        'safe_time': round(safe_time, 1),
        'warning_time': round(warning_time, 1),
        'danger_time': round(danger_time, 1),
        
        'pitch_status': status_label(max_pitch_px, [15, 25]),
        'shoulder_status': status_label(max_shoulder_diff, [10, 20]),
        'roll_status': status_label(max_roll_diff, [4, 8]),
        'fwd_status': status_label(max_fwd_ratio, [1.5, 3]),
        'trunk_status': status_label(max_trunk_angle, [5, 10]),

        # --- 수정 ---: 캡처 시간 데이터 추가
        'pitch_time': round(pitch_capture_time, 1),
        'shoulder_time': round(shoulder_capture_time, 1),
        'roll_time': round(roll_capture_time, 1),
        'fwd_time': round(fwd_capture_time, 1),
        'trunk_time': round(trunk_capture_time, 1)
    })


# (선택) 구버전 수동 baseline API
@app.route('/set_baseline')
def set_baseline():
    global baseline_neck, baseline_shoulder
    if current_neck_center and current_shoulder_center:
        baseline_neck = current_neck_center
        baseline_shoulder = current_shoulder_center
        return "Baseline Set Successfully", 200
    else:
        return "사용자를 감지할 수 없습니다.", 400


if __name__ == '__main__':
    app.run(debug=True)